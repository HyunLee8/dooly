/*
If you wondering where my entire app is 
it's inside components/audio-req.tsx. 
I started working on the component but 
I got lazy and just put everything in one file
but maybe I'll refactor it later.
*/

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
