"""Observed request timings; no inferred speedups or hardware certification."""
import math
import threading
import time
from types import ModuleType


class ExecutionMeasurement:
    def __init__(self, *, cold_start, sample_interval=0.05):
        if type(cold_start) is not bool:
            raise ValueError('explicit cold-start flag required')
        if (type(sample_interval) not in (int, float) or
                not math.isfinite(sample_interval) or sample_interval < 0.01):
            raise ValueError('memory sampling interval must be at least 10 ms')
        self.cold_start = cold_start
        self.interval = sample_interval
        self.started = time.perf_counter()
        self.first_token = self.last_token = None
        self.tentative_tokens = self.events = 0
        self.callback_seconds = 0.0
        self.event_tracking_seconds = 0.0
        self.load_seconds = None
        self._ended = None
        self._outcome = None
        self._committed = 0
        self._peak_rss = None
        self._memory_error = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        try:
            import psutil
            self._process = psutil.Process()
            self._sample_memory()
            if self._peak_rss is not None:
                self._thread = threading.Thread(target=self._memory_loop,
                    name='native-memory-measurement', daemon=True)
                self._thread.start()
        except ImportError:
            self._process = None
            self._memory_error = 'psutil-unavailable'
        except Exception as error:
            self._process = None
            self._memory_error = type(error).__name__

    def _sample_memory(self):
        try:
            value = self._process.memory_info().rss
            with self._lock:
                self._peak_rss = value if self._peak_rss is None else max(value, self._peak_rss)
        except Exception as error:
            self._memory_error = type(error).__name__
            self._stop.set()

    def _memory_loop(self):
        while not self._stop.wait(self.interval):
            self._sample_memory()

    def record_load(self, seconds):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('observed nonnegative loading duration required')
        with self._lock:
            if self._ended is not None:
                raise RuntimeError('measurement is finished')
            self.load_seconds = seconds if self.load_seconds is None else self.load_seconds+seconds

    def event(self, event, callback=None):
        tracked_at = time.perf_counter()
        if type(event) is not dict:
            raise ValueError('structured generation event required')
        count = 1 if event.get('event') == 'generation.token' else 0
        if event.get('event') == 'tokens':
            ids = event.get('token_ids')
            if type(ids) not in (list, tuple) or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError('valid emitted token inventory required')
            count = len(ids)
        with self._lock:
            if self._ended is not None:
                raise RuntimeError('measurement is finished')
            now = time.perf_counter()
            self.events += 1
            if count:
                self.first_token = now if self.first_token is None else self.first_token
                self.last_token = now
                self.tentative_tokens += count
            self.event_tracking_seconds += time.perf_counter()-tracked_at
        if callback is not None:
            started = time.perf_counter()
            try:
                callback(event)
            finally:
                with self._lock:
                    self.callback_seconds += time.perf_counter()-started

    def finish(self, *, outcome, committed_tokens=0):
        if outcome not in ('completed', 'failed', 'cancelled'):
            raise ValueError('explicit measurement outcome required')
        if type(committed_tokens) is not int or committed_tokens < 0:
            raise ValueError('nonnegative committed token count required')
        if outcome != 'completed' and committed_tokens:
            raise ValueError('failed transaction cannot report committed output')
        with self._lock:
            if self._ended is None:
                self._ended = time.perf_counter()
                self._outcome, self._committed = outcome, committed_tokens
            elif (outcome, committed_tokens) != (self._outcome, self._committed):
                raise RuntimeError('measurement outcome cannot change')
        self._stop.set()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join()
        return self.report()

    def report(self):
        with self._lock:
            elapsed = (self._ended or time.perf_counter())-self.started
            decode = None if self.first_token is None or self.last_token is None else self.last_token-self.first_token
            return dict(schema='openagent.execution-measurement.v1', cold_start=self.cold_start,
                outcome=self._outcome or 'running', total_seconds=elapsed,
                loading_seconds=self.load_seconds,
                first_token_seconds=None if self.first_token is None else self.first_token-self.started,
                observed_token_events=self.tentative_tokens, committed_tokens=self._committed,
                request_tokens_per_second=self._committed/elapsed if elapsed and self._outcome == 'completed' else None,
                decode_tokens_per_second=(self._committed-1)/decode if self._outcome == 'completed'
                    and self._committed > 1 and decode else None,
                callback_seconds=self.callback_seconds, event_count=self.events,
                event_tracking_seconds=self.event_tracking_seconds,
                process_peak_rss_bytes=self._peak_rss, memory_error=self._memory_error,
                memory_scope='sampled-whole-process-rss', certification=False)

    def close(self):
        with self._lock:
            outcome = self._outcome or 'cancelled'
            count = self._committed
        return self.finish(outcome=outcome, committed_tokens=count)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def measure_generation(service, session_id, prompt_ids, *, cold_start, **options):
    from .execution_service import ExecutionCancelled
    callback = options.pop('on_event', None)
    measurement = ExecutionMeasurement(cold_start=cold_start)
    try:
        result = service.generate(session_id, prompt_ids,
            on_event=lambda event: measurement.event(event, callback), **options)
    except BaseException as error:
        report = measurement.finish(outcome='cancelled' if isinstance(error, ExecutionCancelled) else 'failed')
        error.execution_measurement = report
        raise
    return result, measurement.finish(outcome='completed', committed_tokens=len(result.token_ids))


def measure_publisher_generation(operator, args, *, cold_start, emit=None, cancelled=None):
    """Observe an explicit CLI operator or an already loaded PublisherSession."""
    from .execution_service import ExecutionCancelled
    if not callable(getattr(operator, 'run', None)):
        raise TypeError('publisher operator with run required')
    measurement = ExecutionMeasurement(cold_start=cold_start)
    try:
        result = operator.run(args, emit=lambda event: measurement.event(event, emit),
                              cancelled=cancelled)
        if type(result) is not dict or type(result.get('generated_ids')) is not list:
            raise ValueError('publisher operator returned no generated token inventory')
        ids = result['generated_ids']
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('publisher operator returned invalid generated tokens')
        load_ms = result.get('load_ms')
        if load_ms is not None and (type(load_ms) not in (int, float) or
                                   not math.isfinite(load_ms) or load_ms < 0):
            raise ValueError('invalid publisher loading observation')
        if load_ms is not None and isinstance(operator, ModuleType):
            measurement.record_load(load_ms/1000)
    except BaseException as error:
        report = measurement.finish(outcome='cancelled' if isinstance(error, ExecutionCancelled) else 'failed')
        error.execution_measurement = report
        raise
    report = measurement.finish(outcome='completed', committed_tokens=len(ids))
    report['operator_reported_load_ms'] = result.get('load_ms')
    return result, report
