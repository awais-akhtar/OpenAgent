"""Bounded text-option scoring shared by native session owners."""
from contextlib import contextmanager
import math

from .failure_reporting import add_failure_note, failure_detail


class ClosedSetScoringMixin:
    """Expose target-model option likelihoods without sampling or committing."""

    def score_options(self, prompt, options, *, normalize='mean', cancelled=None):
        return self.score_option_groups([prompt], [options], [normalize], cancelled=cancelled)[0]

    def score_option_groups(self, prompts, option_groups, normalizations, *, cancelled=None,
                            option_limit=64):
        if (type(prompts) not in (list, tuple) or not prompts or len(prompts) > 64 or
                type(option_groups) not in (list, tuple) or len(option_groups) != len(prompts) or
                type(normalizations) not in (list, tuple) or len(normalizations) != len(prompts) or
                type(option_limit) is not int or option_limit not in (64, 255)):
            raise ValueError('bounded matching Brio prompt, option and normalization groups required')
        if any(type(prompt) is not str or not prompt or len(prompt) > 2 * 1024**2 or
               len(prompt.encode('utf-8')) > 2 * 1024**2 for prompt in prompts):
            raise ValueError('bounded nonempty Brio prompts required')
        if sum(len(prompt.encode('utf-8')) for prompt in prompts) > 8 * 1024**2:
            raise MemoryError('multi-question Brio prompt budget exceeded')
        total_option_chars = total_option_bytes = 0
        for options, normalize in zip(option_groups, normalizations):
            if (normalize not in ('mean', 'sum') or type(options) not in (list, tuple) or
                    not 2 <= len(options) <= option_limit):
                raise ValueError('each Brio group exceeds its option or normalization contract')
            if any(type(option) is not str or not option.strip() or len(option) > 1024**2
                   for option in options) or len(set(options)) != len(options):
                raise ValueError('Brio options must be distinct nonempty bounded text')
            total_option_chars += sum(map(len, options))
            total_option_bytes += sum(len(option.encode('utf-8')) for option in options)
        if total_option_chars > 4 * 1024**2 or total_option_bytes > 4 * 1024**2:
            raise MemoryError('Brio option groups exceed the bounded scoring budget')
        if cancelled is not None and not callable(cancelled):
            raise ValueError('cancellation callback must be callable')
        with self._lock:
            if getattr(self, '_closed', True) or self.service is None or self.session_id is None:
                raise RuntimeError('open native family session required')
            self._artifacts_unchanged()
            prompt_ids = [self.tokenizer.encode(prompt, add_special_tokens=False).ids
                          for prompt in prompts]
            option_ids = [[self.tokenizer.encode(' '+option, add_special_tokens=False).ids
                           for option in options] for options in option_groups]
            if any(not ids for ids in prompt_ids) or any(not ids for group in option_ids for ids in group):
                raise ValueError('prompt and every option must encode to at least one token')
            return self.score_option_groups_token_ids(prompt_ids, option_ids, normalizations,
                option_labels=option_groups, cancelled=cancelled)

    def score_option_groups_token_ids(self, prompt_ids, option_ids, normalizations, *,
                                     option_labels, cancelled=None):
        if (type(prompt_ids) not in (tuple, list) or not prompt_ids or len(prompt_ids) > 64 or
                type(option_ids) not in (tuple, list) or len(option_ids) != len(prompt_ids) or
                type(normalizations) not in (tuple, list) or len(normalizations) != len(prompt_ids) or
                type(option_labels) not in (tuple, list) or len(option_labels) != len(prompt_ids)):
            raise ValueError('bounded matching tokenized prompt and option groups required')
        if any(type(ids) not in (tuple, list) or not ids or
               any(type(token) is not int for token in ids) for ids in prompt_ids):
            raise ValueError('nonempty tokenized prompt suffixes required')
        if any(type(group) not in (tuple, list) or not 2 <= len(group) <= 255 or
               any(type(candidate) not in (tuple, list) or not candidate for candidate in group)
               for group in option_ids) or any(
                   type(labels) not in (tuple, list) or len(labels) != len(candidates) or
                   any(type(label) is not str for label in labels)
                   for labels, candidates in zip(option_labels, option_ids)):
            raise ValueError('bounded tokenized closed sets required')
        if any(normalize not in ('mean', 'sum') for normalize in normalizations):
            raise ValueError('mean/sum normalization required per option group')
        if cancelled is not None and not callable(cancelled):
            raise ValueError('cancellation callback must be callable')
        with self._lock:
            if getattr(self, '_closed', True) or self.service is None or self.session_id is None:
                raise RuntimeError('open native family session required')
            self._artifacts_unchanged()
            from .native_family_cli import _Cancellation
            scored_groups = self.service.score_continuation_groups(
                self.session_id, prompt_ids, option_ids, normalizations=normalizations,
                cancel=_Cancellation(cancelled))
            results = []
            for group_index, (labels, scored) in enumerate(zip(option_labels, scored_groups)):
                choices = [dict(option=labels[index], p=value['probability'],
                    logprob=value['logprob'], mean_logprob=value['mean_logprob'],
                    tokens=value['tokens']) for index, value in enumerate(scored['choices'])]
                choices.sort(key=lambda value: -value['p'])
                entropy = scored['entropy'] / math.log(max(len(labels), 2))
                usage = scored['usage']
                results.append(dict(object='brio.choice', answer=choices[0]['option'],
                    normalize=normalizations[group_index], entropy=min(1.0, max(0.0, entropy)),
                    choices=choices, usage=dict(prompt_tokens=usage['prompt_tokens'],
                        completion_tokens=0, read_tokens=usage['read_tokens'],
                        total_tokens=usage['prompt_tokens']+usage['read_tokens'])))
            return tuple(results)

    @contextmanager
    def scoring_prefix_tokens(self, prefix_ids, *, cancelled=None):
        if (type(prefix_ids) not in (tuple, list) or not prefix_ids or
                any(type(token) is not int for token in prefix_ids)):
            raise ValueError('nonempty shared scoring prefix required')
        if cancelled is not None and not callable(cancelled):
            raise ValueError('cancellation callback must be callable')
        with self._lock:
            if getattr(self, '_closed', True) or self.service is None or self.session_id is None:
                raise RuntimeError('open native family session required')
            self._artifacts_unchanged()
            from .native_family_cli import _Cancellation
            cancel = _Cancellation(cancelled)
            boundary = self.service.snapshot(self.session_id)
            primary = None
            try:
                self.service.prefill(self.session_id, prefix_ids, cancel=cancel)
                yield
            except BaseException as error:
                primary = error
                raise
            finally:
                cleanup = None
                try:
                    self.service.restore(self.session_id, boundary)
                except BaseException as error:
                    cleanup = error
                    if primary is not None:
                        add_failure_note(primary, 'scoring prefix rollback failed: ' +
                                         failure_detail(error))
                try:
                    self.service.release_snapshot(self.session_id, boundary)
                except BaseException as error:
                    if cleanup is None:
                        cleanup = error
                        if primary is not None:
                            add_failure_note(primary,
                                'scoring prefix snapshot release failed: ' + failure_detail(error))
                    else:
                        add_failure_note(primary or cleanup,
                            'scoring prefix snapshot release failed: ' + failure_detail(error))
                if primary is None and cleanup is not None:
                    raise cleanup
