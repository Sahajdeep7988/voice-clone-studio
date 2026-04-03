import { AppShell } from "@/components/AppShell";
import { ActiveTraining } from "@/components/screens/ActiveTraining";

export default function TrainingPage({ params }: { params: { id: string } }) {
  return (
    <AppShell>
      <ActiveTraining sessionId={params.id} />
    </AppShell>
  );
}
