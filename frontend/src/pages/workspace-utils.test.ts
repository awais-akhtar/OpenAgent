import { describe, expect, it } from 'vitest';
import { createWorkspaceDocumentPayload, updateWorkspaceDocumentPayload, updateWorkspaceItemPayload, workspaceLibraryPaths, workspacePreviewBlocks } from './workspace-utils';

describe('project workspace contracts', () => {
  it('keeps both file libraries scoped to the active project and search query', () => {
    expect(workspaceLibraryPaths('team/alpha', 'release notes')).toEqual({
      workspace: '/api/workspace/items?project=team%2Falpha&q=release%20notes&limit=100',
      documents: '/api/documents/library?project=team%2Falpha&q=release%20notes&limit=100',
    });
  });

  it('preserves the document version precondition and existing item update shape', () => {
    expect(createWorkspaceDocumentPayload('Plan', '# Plan', 'markdown', 'alpha')).toEqual({ title: 'Plan', content: '# Plan', format: 'markdown', project: 'alpha' });
    expect(updateWorkspaceDocumentPayload('Plan', '# Revised', 4)).toEqual({ title: 'Plan', content: '# Revised', if_version: 4 });
    expect(updateWorkspaceItemPayload('item-9', 'Plan', 'Text')).toEqual({ item_id: 'item-9', title: 'Plan', content: 'Text' });
  });

  it('builds a safe semantic Markdown preview from headings, lists, and fenced code', () => {
    expect(workspacePreviewBlocks('# Notes\nA paragraph.\n- One\n```ts\nconst x = 1;\n```')).toEqual([
      { kind: 'heading', level: 1, text: 'Notes' },
      { kind: 'paragraph', text: 'A paragraph.' },
      { kind: 'list', text: 'One' },
      { kind: 'code', text: 'const x = 1;' },
    ]);
  });
});
