import * as SeparatorPrimitive from '@radix-ui/react-separator';
import type { ComponentProps } from 'react';

export function Separator({ className = '', ...props }: ComponentProps<typeof SeparatorPrimitive.Root>) {
  return <SeparatorPrimitive.Root className={`ui-separator ${className}`.trim()} {...props} />;
}
