export type WorkspacePreviewBlock = { kind: 'heading' | 'paragraph' | 'list' | 'code'; text: string; level?: number };
// Document import accepts JSON (base64 content), and the Python JSON request
// body limit defaults to 1 MiB. Keep headroom for encoding and metadata.
export const WORKSPACE_IMPORT_MAX_BYTES = 700 * 1024;

export function workspaceImportRoute(filename: string) {
  return /\.pdf$/i.test(filename) ? '/api/documents/import-pdf' : '/api/documents/import';
}

export async function workspaceImportPayload(file: File, project: string) {
  if (file.size > WORKSPACE_IMPORT_MAX_BYTES) throw new Error('Choose a file smaller than 700 KB. This import route uses the service JSON request limit, which may be configured lower.');
  const filename = file.name.split(/[\\/]/).pop()?.trim() || 'imported-document.txt';
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return { filename, title: filename.replace(/\.[^.]+$/, ''), project, content_base64: btoa(binary) };
}

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
