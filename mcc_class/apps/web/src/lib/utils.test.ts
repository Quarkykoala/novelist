import { describe, expect, it } from 'vitest';
import { cn } from './utils';

describe('cn', () => {
  it('merges classnames and drops falsy values', () => {
    const enabled = false;
    expect(cn('alpha', enabled && 'beta', 'gamma')).toBe('alpha gamma');
  });

  it('keeps the last conflicting Tailwind class', () => {
    expect(cn('p-2', 'p-4')).toBe('p-4');
  });
});
