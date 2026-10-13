import { cloneElement, type ReactElement, type ReactNode } from 'react';

/** Tooltips use the browser's native title behavior; labels remain available to assistive technology. */
export function TooltipProvider({ children }: { children: ReactNode }) { return <>{children}</>; }

export function Tooltip({ label, children }: { label: string; children: ReactElement }) {
  // Native title is intentionally used here: it needs no global portal or runtime.
  return cloneElement(children, { title: label } as Partial<unknown>);
}
