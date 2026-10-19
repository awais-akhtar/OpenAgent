import {
  Activity, Bot, BrainCircuit, CalendarDays, Code2, FileText, FlaskConical,
  Headphones, Home, MessageSquare, Network, Search, Send,
  Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Users, Workflow,
  Wrench, Vault, Waypoints, type IconComponent,
} from './icons';

export type ViewId = 'dashboard' | 'chat' | 'codebase' | 'voice' | 'research' | 'agents' | 'knowledge' | 'workspace' | 'automation' | 'assistant' | 'communications' | 'models' | 'security' | 'developer' | 'settings' | 'engine' | 'science' | 'calendar' | 'twin' | 'vault' | 'scraper' | 'finetune';
export type NavItem = { id: ViewId; label: string; icon: IconComponent; keywords?: string };
export type NavGroup = { label: string; key: string; items: NavItem[] };

/** Primary tasks stay visible; the full set remains reachable from Search. */
export const groups: NavGroup[] = [
  { label: 'Your workspace', key: 'workspace', items: [
    { id: 'chat', label: 'Chat', icon: MessageSquare, keywords: 'new conversation' },
    { id: 'workspace', label: 'Projects', icon: FileText, keywords: 'documents files' },
    { id: 'knowledge', label: 'Knowledge', icon: BrainCircuit },
    { id: 'automation', label: 'Automations', icon: Workflow, keywords: 'workflows scheduled tasks' },
    { id: 'agents', label: 'Agents', icon: Bot, keywords: 'assistants registry marketplace' },
    { id: 'settings', label: 'Settings', icon: Settings2, keywords: 'preferences configuration' },
  ] },
  { label: 'More tools', key: 'advanced', items: [
    { id: 'dashboard', label: 'Activity', icon: Home, keywords: 'overview status dashboard' },
    { id: 'assistant', label: 'Personal assistant', icon: Sparkles, keywords: 'tasks goals coaching' },
    { id: 'research', label: 'Research', icon: Search },
    { id: 'codebase', label: 'Codebase', icon: Code2, keywords: 'repository project' },
    { id: 'voice', label: 'Voice', icon: Headphones },
    { id: 'calendar', label: 'Calendar', icon: CalendarDays },
    { id: 'communications', label: 'Communications', icon: Send, keywords: 'inbox messages' },
    { id: 'models', label: 'Models', icon: SlidersHorizontal },
    { id: 'security', label: 'Security', icon: ShieldCheck },
    { id: 'developer', label: 'Developer', icon: Wrench, keywords: 'integrations api' },
    { id: 'engine', label: 'Engine', icon: Activity, keywords: 'adaptive' },
    { id: 'science', label: 'Scientific AI', icon: FlaskConical },
    { id: 'scraper', label: 'Web Scraper', icon: Network },
    { id: 'finetune', label: 'Fine-tuning', icon: Waypoints },
    { id: 'twin', label: 'Personal Context', icon: Users, keywords: 'digital twin' },
    { id: 'vault', label: 'Vault', icon: Vault, keywords: 'credentials secrets' },
  ] },
];

export const views = groups.flatMap(group => group.items);
export const titles = Object.fromEntries(views.map(view => [view.id, view.label])) as Record<ViewId, string>;
export function isViewId(value: string | null): value is ViewId { return !!value && views.some(view => view.id === value); }

/** New sessions open Chat; every declared view remains directly addressable. */
export function migrateLegacyView(value: string | null): ViewId {
  if (!value) return 'chat';
  return isViewId(value) ? value : 'chat';
}

export const primaryNavigation: NavItem[] = [
  { id: 'dashboard', label: 'Home', icon: Home, keywords: 'overview activity workspace' },
  { id: 'codebase', label: 'CodeQ', icon: Code2, keywords: 'repository coding workspace' },
  { id: 'agents', label: 'Agents', icon: Bot },
  { id: 'research', label: 'Research', icon: Search },
  { id: 'workspace', label: 'Files', icon: FileText, keywords: 'projects documents' },
  { id: 'automation', label: 'Automations', icon: Workflow, keywords: 'tasks workflows' },
  { id: 'calendar', label: 'Calendar', icon: CalendarDays },
  { id: 'models', label: 'Models', icon: SlidersHorizontal },
];
