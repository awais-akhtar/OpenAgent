import type { ComponentProps } from 'react';

type SeparatorProps = Omit<ComponentProps<'div'>, 'role'> & { orientation?: 'horizontal' | 'vertical'; decorative?: boolean };
export function Separator({ className = '', orientation = 'horizontal', decorative = true, ...props }: SeparatorProps) {
  return <div {...props} className={`ui-separator ${className}`.trim()} role={decorative ? undefined : 'separator'} aria-orientation={decorative ? undefined : orientation} data-orientation={orientation} />;
}
