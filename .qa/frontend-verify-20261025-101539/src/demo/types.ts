export type DemoRecord = Record<string, unknown> & { id?: string; name?: string; title?: string; status?: string };
export type DemoOperation = 'create' | 'update' | 'delete' | 'run' | 'toggle' | 'search' | 'reset';
