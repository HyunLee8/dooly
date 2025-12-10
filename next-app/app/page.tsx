import AudioReq from "@/components/audio-req";

export default function Home() {
  return (
      <div className="flex flex-col min-h-screen w-full items-center justify-center font-sans" style={{ position: 'relative', zIndex: 1 }}>
        <div className="gap-5 flex flex-col items-center">
          <AudioReq />
        </div>
      </div>
  );
}
