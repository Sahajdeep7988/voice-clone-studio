import { AppShell } from "@/components/AppShell";
import { Curation } from "@/components/screens/Curation";

export default function CurationPage({ params }: { params: { id: string } }) {
  return (
    <AppShell>
      <Curation sessionId={params.id} />
    </AppShell>
  );
}
