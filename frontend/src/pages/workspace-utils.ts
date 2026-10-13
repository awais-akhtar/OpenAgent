export type WorkspacePreviewBlock = { kind: 'heading' | 'paragraph' | 'list' | 'code'; text: string; level?: number };

export function workspaceLibraryPaths(project: string, query: string) {
  const scope = encodeURIComponent(project);
  const search = encodeURIComponent(query);
  return {
    workspace: `/api/workspace/items?project=${scope}&q=${search}&limit=100`,
    documents: `/api/documents/library?project=${scope}&q=${search}&limit=100`,
  };
}

export function createWorkspaceDocumentPayload(title: string, content: string, format: string, project: string) {
  return { title, content, format, project };
}

export function updateWorkspaceDocumentPayload(title: string, content: string, version: number) {
  return { title, content, if_version: version };
}

export function updateWorkspaceItemPayload(itemId: unknown, title: string, content: string) {
  return { item_id: itemId, title, content };
}

export function workspacePreviewBlocks(content: string): WorkspacePreviewBlock[] {
  const blocks: WorkspacePreviewBlock[] = [];
  let code = false;
  for (const line of content.split(/\r?\n/)) {
    if (/^```/.test(line.trim())) { code = !code; continue; }
    if (code) { const last = blocks.at(-1); if (last?.kind === 'code') last.text += `\n${line}`; else blocks.push({ kind: 'code', text: line }); continue; }
    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) blocks.push({ kind: 'heading', level: heading[1].length, text: heading[2] });
    else if (/^\s*[-*+]\s+/.test(line)) blocks.push({ kind: 'list', text: line.replace(/^\s*[-*+]\s+/, '') });
    else if (line.trim()) blocks.push({ kind: 'paragraph', text: line });
  }
  return blocks;
}
