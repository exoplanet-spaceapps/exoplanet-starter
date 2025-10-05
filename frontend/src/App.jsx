import React, { useEffect, useState } from "react";
import Navbar from "./components/Navbar/Navbar";
import Hero from "./components/Hero/Hero.jsx";
import HeroCard from "./components/HeroCard/HeroCard.jsx";
import planetVideo from "./assets/GSFC_20190627_TESS_m13223_L98-59b_Full_Rotation~orig.mp4";
import Rapidscat from "./components/Rapidscat/Rapidscat.jsx";
import Satelite from "./components/Satelite/Satelite.jsx";
import Footer5 from "./components/Footer/Footer5.jsx";
import DataPortal from "./components/DataPortal/DataPortal.jsx";
import AOS from "aos";
import "aos/dist/aos.css";

const App = () => {
  const [view, setView] = useState("landing");

  useEffect(() => {
    AOS.init({
      duration: 1200,
      easing: "ease-in-out",
    });
  }, []);

  const showLanding = view === "landing";
  return (
    <div className="">
      <div className="h-[700px] relative">
        <video
          autoPlay
          loop
          muted
          className="fixed right-0 top-0 h-[700px] w-full object-cover z-[-1]"
        >
          <source src={planetVideo} type="video/mp4" />
        </video>
        <Navbar />
        {showLanding ? (
          <Hero onGetStarted={() => setView("data")} />
        ) : (
          <DataPortal onBack={() => setView("landing")} />
        )}
      </div>
      {showLanding && (
        <>
          <HeroCard />
          <Rapidscat />
          <Satelite />
        </>
      )}
      <Footer5 />
    </div>
  );
};

export default App;
