import {
  Activity, Bot, BrainCircuit, CalendarDays, Code2, FileText, FlaskConical,
  FolderGit2, Headphones, Home, MessageSquare, Network, Search, Send,
  Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Users, Workflow,
  Wrench, Vault, Waypoints, type IconComponent,
} from './icons';

export type ViewId = 'dashboard' | 'chat' | 'codebase' | 'voice' | 'research' | 'phd' | 'agents' | 'knowledge' | 'workspace' | 'automation' | 'assistant' | 'communications' | 'models' | 'security' | 'developer' | 'settings' | 'engine' | 'science' | 'calendar' | 'twin' | 'vault' | 'scraper' | 'finetune';
export type NavItem = { id: ViewId; label: string; icon: IconComponent; keywords?: string };
export type NavGroup = { label: string; key: string; items: NavItem[] };

/** All legacy OpenAgent screens remain discoverable from the Tools launcher. */
export const groups: NavGroup[] = [
  { label: 'Home and assistant', key: 'home', items: [
    { id: 'dashboard', label: 'Activity', icon: Home, keywords: 'overview status dashboard' },
    { id: 'chat', label: 'Chat', icon: MessageSquare, keywords: 'new conversation' },
    { id: 'assistant', label: 'Assistant', icon: Sparkles, keywords: 'tasks goals coaching' },
    { id: 'agents', label: 'Assistants', icon: Bot, keywords: 'agent registry marketplace' },
  ] },
  { label: 'Create and explore', key: 'create', items: [
    { id: 'research', label: 'Research', icon: Search },
    { id: 'workspace', label: 'Workspace', icon: FileText, keywords: 'documents files' },
    { id: 'knowledge', label: 'Knowledge', icon: BrainCircuit },
    { id: 'codebase', label: 'CodeQ', icon: Code2, keywords: 'codebase project' },
    { id: 'voice', label: 'Voice', icon: Headphones },
    { id: 'phd', label: 'Application Studio', icon: FolderGit2, keywords: 'phd applications' },
    { id: 'science', label: 'Scientific AI', icon: FlaskConical },
    { id: 'scraper', label: 'Web Scraper', icon: Network },
    { id: 'finetune', label: 'Fine-tuning', icon: Waypoints },
  ] },
  { label: 'Run and coordinate', key: 'run', items: [
    { id: 'automation', label: 'Automations', icon: Workflow, keywords: 'workflows scheduled tasks' },
    { id: 'calendar', label: 'Calendar', icon: CalendarDays },
    { id: 'communications', label: 'Communications', icon: Send, keywords: 'inbox messages' },
    { id: 'engine', label: 'Adaptive Engine', icon: Activity },
  ] },
  { label: 'Models and privacy', key: 'system', items: [
    { id: 'models', label: 'Models', icon: SlidersHorizontal },
    { id: 'security', label: 'Security', icon: ShieldCheck },
    { id: 'developer', label: 'Developer', icon: Wrench, keywords: 'integrations api' },
    { id: 'twin', label: 'Personal Context', icon: Users, keywords: 'digital twin' },
    { id: 'vault', label: 'Vault', icon: Vault, keywords: 'credentials secrets' },
    { id: 'settings', label: 'Settings', icon: Settings2, keywords: 'preferences configuration' },
  ] },
];

export const views = groups.flatMap(group => group.items);
export const titles = Object.fromEntries(views.map(view => [view.id, view.label])) as Record<ViewId, string>;
export function isViewId(value: string | null): value is ViewId { return !!value && views.some(view => view.id === value); }

/** Home is conversation-first; old dashboard links resolve to the new home. */
export function migrateLegacyView(value: string | null): ViewId {
  if (!value) return 'chat';
  if (value === 'dashboard') return 'chat';
  return isViewId(value) ? value : 'chat';
}

export const primaryNavigation: NavItem[] = [
  { id: 'agents', label: 'Assistants', icon: Bot },
  { id: 'automation', label: 'Scheduled', icon: CalendarDays, keywords: 'tasks automations' },
];
