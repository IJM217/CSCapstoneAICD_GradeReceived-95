//utils.js: utility functions for combining CSS classes and other common operations
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge"

//utility function that combines clsx and tailwind-merge for optimal class name handling
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
