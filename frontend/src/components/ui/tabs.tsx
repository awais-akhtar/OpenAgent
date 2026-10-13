import { createContext, useContext, useId, type ComponentProps, type HTMLAttributes, type ReactNode } from 'react';

type TabsContextValue = { value: string; setValue: (value: string) => void; id: string };
const TabsContext = createContext<TabsContextValue | null>(null);

export function Tabs({ value, onValueChange, children, ...props }: Omit<ComponentProps<'div'>, 'value' | 'onChange'> & { value: string; onValueChange: (value: string) => void; children: ReactNode }) {
  const id = useId();
  return <TabsContext.Provider value={{ value, setValue: onValueChange, id }}><div {...props}>{children}</div></TabsContext.Provider>;
}

export function TabsList({ className = '', ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div role="tablist" className={`ui-tabs-list ${className}`.trim()} {...props} />;
}

export function TabsTrigger({ value, className = '', children, onClick, onKeyDown, ...props }: ComponentProps<'button'> & { value: string }) {
  const tabs = useContext(TabsContext);
  if (!tabs) throw new Error('TabsTrigger must be rendered inside Tabs.');
  const selected = tabs.value === value;
  const ownId = `${tabs.id}-tab-${value.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
  const panelId = `knowledge-panel-${value}`;
  const select = () => tabs.setValue(value);
  return <button {...props} type="button" role="tab" id={props.id ?? ownId} aria-selected={selected} aria-controls={props['aria-controls'] ?? panelId} tabIndex={selected ? 0 : -1} data-state={selected ? 'active' : 'inactive'} className={`ui-tabs-trigger oa-focus-ring ${className}`.trim()} onClick={event => { onClick?.(event); if (!event.defaultPrevented) select(); }} onKeyDown={event => {
    onKeyDown?.(event);
    if (event.defaultPrevented || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const list = event.currentTarget.closest('[role="tablist"]');
    const items = list ? [...list.querySelectorAll<HTMLButtonElement>('[role="tab"]')] : [];
    const index = items.indexOf(event.currentTarget);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + items.length) % items.length;
    event.preventDefault(); items[next]?.focus(); items[next]?.click();
  }}>{children}</button>;
}
