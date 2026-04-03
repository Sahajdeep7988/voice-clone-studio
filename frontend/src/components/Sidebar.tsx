"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Plus,
  Grid3x3,
  Mic2,
  Settings,
} from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/session-builder", label: "New Training", icon: Plus },
  { href: "/my-models", label: "My Models", icon: Grid3x3 },
  { href: "/inference", label: "Inference", icon: Mic2 },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 border-subtle border-r bg-card">
      {/* Logo */}
      <div className="flex items-center gap-3 border-subtle border-b px-6 py-6">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-amber">
          <span className="text-sm font-bold text-background">VC</span>
        </div>
        <div>
          <h1 className="text-lg font-semibold">Voice Clone</h1>
          <p className="text-xs text-foreground/60">Studio</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-1 px-4 py-6">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-4 py-3 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-accent-amber text-background"
                  : "text-foreground/70 hover:bg-card-muted hover:text-foreground"
              }`}
            >
              <Icon size={20} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 border-subtle border-t bg-card px-6 py-4">
        <p className="text-xs text-foreground/50">v0.1.0</p>
        <p className="text-xs text-foreground/40">Local Voice Cloning</p>
      </div>
    </aside>
  );
}
