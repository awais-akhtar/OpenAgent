import { Children, createContext, isValidElement, useContext, useEffect, useId, useRef, type ComponentProps, type HTMLAttributes, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

type DialogContextValue = { open: boolean; onOpenChange?: (open: boolean) => void; titleId?: string; descriptionId?: string };
const DialogContext = createContext<DialogContextValue>({ open: false });

export function Dialog({ open, onOpenChange, children }: { open: boolean; onOpenChange?: (open: boolean) => void; children: ReactNode }) {
  const id = useId();
  return <DialogContext.Provider value={{ open, onOpenChange, titleId: `${id}-title`, descriptionId: `${id}-description` }}>{children}</DialogContext.Provider>;
}

export function DialogTrigger(props: ComponentProps<'button'>) {
  const context = useContext(DialogContext);
  return <button type="button" {...props} onClick={event => { props.onClick?.(event); if (!event.defaultPrevented) context.onOpenChange?.(true); }} />;
}
export function DialogClose(props: ComponentProps<'button'>) {
  const context = useContext(DialogContext);
  return <button type="button" {...props} onClick={event => { props.onClick?.(event); if (!event.defaultPrevented) context.onOpenChange?.(false); }} />;
}
export function DialogTitle(props: HTMLAttributes<HTMLHeadingElement>) { const { titleId } = useContext(DialogContext); return <h2 data-dialog-title="" id={props.id ?? titleId} {...props} />; }
export function DialogDescription(props: HTMLAttributes<HTMLParagraphElement>) { const { descriptionId } = useContext(DialogContext); return <p data-dialog-description="" id={props.id ?? descriptionId} {...props} />; }

function containsComponent(node: ReactNode, component: unknown): boolean {
  return Children.toArray(node).some(child => isValidElement(child) && (child.type === component || containsComponent((child.props as { children?: ReactNode }).children, component)));
}

export function DialogContent({ className = '', children, onClick, onKeyDown, ...props }: ComponentProps<'div'>) {
  const context = useContext(DialogContext);
  const contentRef = useRef<HTMLDivElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!context.open) return;
    previouslyFocused.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    contentRef.current?.focus();
    return () => {
      document.body.style.overflow = previousOverflow;
      previouslyFocused.current?.focus();
    };
  }, [context.open]);

  if (!context.open || typeof document === 'undefined') return null;
  const closeIfOutside = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) context.onOpenChange?.(false);
    onClick?.(event);
  };
  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    onKeyDown?.(event);
    if (event.defaultPrevented) return;
    if (event.key === 'Escape') { event.preventDefault(); context.onOpenChange?.(false); return; }
    if (event.key !== 'Tab' || !contentRef.current) return;
    const items = [...contentRef.current.querySelectorAll<HTMLElement>('a[href],button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex]:not([tabindex="-1"])')].filter(item => item.getAttribute('aria-hidden') !== 'true');
    if (!items.length) { event.preventDefault(); contentRef.current.focus(); return; }
    const first = items[0]; const last = items[items.length - 1];
    if (event.shiftKey && (document.activeElement === first || document.activeElement === contentRef.current)) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  };
  const labelledBy = props['aria-labelledby'] ?? (containsComponent(children, DialogTitle) ? context.titleId : undefined);
  const describedBy = props['aria-describedby'] ?? (containsComponent(children, DialogDescription) ? context.descriptionId : undefined);
  const content = <div className="ui-dialog-overlay" onMouseDown={closeIfOutside}>
    <div {...props} ref={contentRef} role="dialog" aria-modal="true" aria-labelledby={labelledBy} aria-describedby={describedBy} className={`ui-dialog-content ${className}`.trim()} tabIndex={-1} onKeyDown={handleKeyDown}>
      {children}
    </div>
  </div>;
  return createPortal(content, document.body);
}
