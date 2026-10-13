import type { ButtonHTMLAttributes } from 'react';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'primary' | 'destructive' | 'ghost';
  size?: 'default' | 'sm' | 'icon';
};

const variantClass = { default: '', primary: 'primary', destructive: 'danger', ghost: 'quiet' } as const;
const sizeClass = { default: '', sm: 'button-sm', icon: 'icon-button' } as const;

/** Small app-owned shadcn-style button surface; styles remain in the local design system. */
export function Button({ variant = 'default', size = 'default', className = '', ...props }: ButtonProps) {
  return <button className={`btn ${variantClass[variant]} ${sizeClass[size]} ${className}`.trim()} {...props} />;
}
