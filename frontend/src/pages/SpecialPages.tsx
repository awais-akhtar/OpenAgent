import type { ReactNode } from 'react';
import { CodebasePage } from './CodebasePage';
import { KnowledgePage } from './KnowledgePage';
import { ModelsPage } from './ModelsPage';
import { ResearchPage } from './ResearchPage';
import { SecurityPage } from './SecurityPage';
import { SettingsPage } from './SettingsPage';
import { VaultPage } from './VaultPage';
import { VoicePage } from './VoicePage';
import { WorkspacePage } from './WorkspacePage';
export { specialPageIds } from './specialPageIds';

const pages: Record<string, () => ReactNode> = {
  codebase: CodebasePage,
  knowledge: KnowledgePage,
  models: ModelsPage,
  research: ResearchPage,
  security: SecurityPage,
  settings: SettingsPage,
  vault: VaultPage,
  voice: VoicePage,
  workspace: WorkspacePage,
};

export function SpecialPages({ id }: { id: string }) {
  const Page = pages[id];
  return Page ? <Page /> : null;
}

