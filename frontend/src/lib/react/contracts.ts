export type ResourceRead = { label: string; path: string; key?: string };
export type ResourceField = { name: string; label: string; type?: 'text' | 'textarea' | 'number' | 'password' | 'select'; options?: string[]; optionsFrom?: string; required?: boolean; hint?: string };
export type ResourceAction = { label: string; path: (values: Record<string, string>) => `/api/${string}`; fields: ResourceField[]; payload: (values: Record<string, string>) => unknown; method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'; confirmText?: string };
