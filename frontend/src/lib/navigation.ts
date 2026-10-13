import { Activity, Bot, BrainCircuit, CalendarDays, Code2, FileText, FlaskConical, FolderGit2, Headphones, Home, MessageSquare, Network, Search, Send, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Users, Workflow, Wrench, Vault, Waypoints, type IconComponent } from './icons';

export type ViewId = 'dashboard' | 'chat' | 'codebase' | 'voice' | 'research' | 'phd' | 'agents' | 'knowledge' | 'workspace' | 'automation' | 'assistant' | 'communications' | 'models' | 'security' | 'developer' | 'settings' | 'engine' | 'science' | 'calendar' | 'twin' | 'vault' | 'scraper' | 'finetune';
export type NavItem = { id: ViewId; label: string; icon: IconComponent; keywords?: string };
export type NavGroup = { label: string; key: string; items: NavItem[] };
export const groups: NavGroup[] = [
  { label: 'Home', key: 'home', items: [{ id: 'dashboard', label: 'Overview', icon: Home }, { id: 'chat', label: 'Chat', icon: MessageSquare }, { id: 'assistant', label: 'Assistant', icon: Sparkles }] },
  { label: 'Create & research', key: 'create', items: [{ id: 'research', label: 'Research', icon: Search }, { id: 'workspace', label: 'Documents', icon: FileText }, { id: 'knowledge', label: 'Knowledge', icon: BrainCircuit }, { id: 'codebase', label: 'CodeQ', icon: Code2 }, { id: 'voice', label: 'Voice', icon: Headphones }] },
  { label: 'Build & automate', key: 'build', items: [{ id: 'agents', label: 'Agents', icon: Bot }, { id: 'automation', label: 'Workflows', icon: Workflow }, { id: 'calendar', label: 'Calendar', icon: CalendarDays }, { id: 'communications', label: 'Communications', icon: Send }, { id: 'scraper', label: 'Web scraper', icon: Network }] },
  { label: 'Models & compute', key: 'compute', items: [{ id: 'models', label: 'Models', icon: SlidersHorizontal }, { id: 'engine', label: 'Adaptive Engine', icon: Activity }, { id: 'finetune', label: 'Fine-tuning', icon: Waypoints }, { id: 'science', label: 'Scientific AI', icon: FlaskConical }] },
  { label: 'Private data', key: 'private', items: [{ id: 'twin', label: 'Personal Context', icon: Users }, { id: 'vault', label: 'Vault', icon: Vault }, { id: 'security', label: 'Security', icon: ShieldCheck }] },
  { label: 'Studios', key: 'studio', items: [{ id: 'phd', label: 'Application Studio', icon: FolderGit2 }] },
  { label: 'System', key: 'system', items: [{ id: 'developer', label: 'Developer', icon: Wrench }, { id: 'settings', label: 'Settings', icon: Settings2 }] },
];
export const views = groups.flatMap(group => group.items);
export const titles = Object.fromEntries(views.map(view => [view.id, view.label])) as Record<ViewId, string>;
export function isViewId(value: string | null): value is ViewId { return !!value && views.some(view => view.id === value); }
