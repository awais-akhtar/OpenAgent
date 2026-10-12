export type StateListener<T> = (state: Readonly<T>, changedKeys: ReadonlySet<keyof T>) => void;

export interface StateStore<T extends Record<string, unknown>> {
  get(): Readonly<T>;
  patch(update: Partial<T>): Readonly<T>;
  subscribe(listener: StateListener<T>): () => void;
}

export function createStateStore<T extends Record<string, unknown>>(initial: T): StateStore<T> {
  let current = { ...initial };
  const listeners = new Set<StateListener<T>>();

  return {
    get: () => current,
    patch(update) {
      const changedKeys = new Set<keyof T>();
      for (const key of Object.keys(update) as Array<keyof T>) {
        if (!Object.is(current[key], update[key])) changedKeys.add(key);
      }
      if (!changedKeys.size) return current;
      current = { ...current, ...update };
      listeners.forEach(listener => listener(current, changedKeys));
      return current;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
