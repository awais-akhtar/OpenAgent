import * as monaco from 'monaco-editor/esm/vs/editor/editor.api.js';
import { loader } from '@monaco-editor/react';
import 'monaco-editor/esm/vs/language/typescript/monaco.contribution.js';
// Monaco's TypeScript language service provides diagnostics and editor actions,
// while the basic-language contribution provides the synchronous Monarch
// tokenizer used to paint syntax tokens in the CodeQ surface.
import 'monaco-editor/esm/vs/basic-languages/typescript/typescript.contribution.js';
import 'monaco-editor/esm/vs/basic-languages/javascript/javascript.contribution.js';
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
import CssWorker from 'monaco-editor/esm/vs/language/css/css.worker?worker';
import HtmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker';
import TypeScriptWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';

type MonacoEnvironment = { getWorker: (_workerId: string, label: string) => Worker };
const workerGlobal = globalThis as typeof globalThis & { MonacoEnvironment?: MonacoEnvironment };

workerGlobal.MonacoEnvironment = {
  getWorker(_workerId, label) {
    if (label === 'json') return new JsonWorker();
    if (label === 'css' || label === 'scss' || label === 'less') return new CssWorker();
    if (label === 'html' || label === 'handlebars' || label === 'razor') return new HtmlWorker();
    if (label === 'typescript' || label === 'javascript') return new TypeScriptWorker();
    return new EditorWorker();
  },
};

// Use the checked-in monaco-editor package instead of the CDN default.
// This keeps CodeQ's editor usable offline and avoids external script/worker requests.
loader.config({ monaco });

// Keep Monaco's editor surface and token contrast aligned with the standalone
// approved CodeQ sample. In particular, the source highlights language keywords
// in violet while leaving the rest of the code in the workspace text color.
monaco.editor.defineTheme('openagent-approved-light', {
  base: 'vs',
  inherit: true,
  rules: [
    { token: '', foreground: '142039' },
    { token: 'keyword', foreground: 'A34DE5' },
    { token: 'comment', foreground: '6E7C96' },
  ],
  colors: {
    'editor.background': '#FFFFFF',
    'editor.foreground': '#142039',
    'editorLineNumber.foreground': '#94A0B3',
    'editorLineNumber.activeForeground': '#94A0B3',
    'editorGutter.background': '#FFFFFF',
  },
});

monaco.editor.defineTheme('openagent-approved-dark', {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: '', foreground: 'EDF3FE' },
    { token: 'keyword', foreground: 'A34DE5' },
    { token: 'comment', foreground: 'A1B0C9' },
  ],
  colors: {
    'editor.background': '#151F31',
    'editor.foreground': '#EDF3FE',
    'editorLineNumber.foreground': '#8194B2',
    'editorLineNumber.activeForeground': '#8194B2',
    'editorGutter.background': '#151F31',
  },
});

