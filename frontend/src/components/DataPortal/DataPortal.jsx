import React, { useRef, useState } from "react";

const DataPortal = ({ onBack }) => {
  const fileInputRef = useRef(null);
  const [uploadState, setUploadState] = useState({ status: "idle" });

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    if (
      file.type !== "text/csv" &&
      !file.name.toLowerCase().endsWith(".csv")
    ) {
      setUploadState({
        status: "error",
        message: "Please choose a .csv file.",
      });
      return;
    }

    setUploadState({ status: "reading" });

    const reader = new FileReader();

    reader.onerror = () => {
      setUploadState({
        status: "error",
        message: "We could not read that file. Please try again.",
      });
    };

    reader.onload = (loadEvent) => {
      const text = loadEvent.target?.result ?? "";
      const preview = text.split(/\r?\n/).slice(0, 5).join("\n");
      const rows = text
        .split(/\r?\n/)
        .map((row) => row.trim())
        .filter((row) => row.length > 0);

      setUploadState({
        status: "success",
        fileName: file.name,
        rowCount: rows.length,
        preview,
      });
    };

    reader.readAsText(file);
  };

  return (
    <section className="h-[700px] flex items-center justify-center px-6 py-24 text-white relative">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,text/csv"
        onChange={handleFileChange}
        className="sr-only"
      />
      <div className="bg-black/50 border border-white/10 rounded-2xl max-w-3xl w-full p-8 backdrop-blur-md space-y-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <h2 className="text-3xl font-semibold" data-aos="fade-up">
            Bring Your Own Data
          </h2>
          <button
            onClick={onBack}
            className="rounded-md border border-white/40 px-4 py-2 text-sm font-medium text-white/80 hover:text-white hover:border-white/70 transition"
            data-aos="fade-up"
            data-aos-delay="100"
          >
            Back to Overview
          </button>
        </div>
        <p className="text-white/80 max-w-2xl" data-aos="fade-up" data-aos-delay="150">
          Upload a CSV from your own observations to explore orbital patterns and transit signals.
        </p>
        <div className="grid gap-4 sm:grid-cols-2" data-aos="fade-up" data-aos-delay="200">
          <button
            onClick={handleUploadClick}
            className="bg-blue-400 text-white hover:bg-blue-500 px-4 py-3 rounded-lg font-semibold transition"
          >
            Upload CSV
          </button>
          <button
            disabled
            className="bg-white/10 text-white/80 px-4 py-3 rounded-lg font-semibold border border-white/20 cursor-not-allowed"
          >
            Try Sample Data (Coming Soon)
          </button>
        </div>
        {uploadState.status === "reading" && (
          <div
            className="bg-blue-500/10 border border-blue-400/30 text-blue-100 rounded-lg p-4"
            data-aos="fade-up"
            data-aos-delay="250"
          >
            Reading your file...
          </div>
        )}
        {uploadState.status === "error" && (
          <div
            className="bg-red-500/10 border border-red-400/40 text-red-100 rounded-lg p-4"
            data-aos="fade-up"
            data-aos-delay="250"
          >
            {uploadState.message}
          </div>
        )}
        {uploadState.status === "success" && (
          <div
            className="bg-emerald-500/10 border border-emerald-400/40 text-emerald-100 rounded-lg p-4 space-y-3"
            data-aos="fade-up"
            data-aos-delay="250"
          >
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <p className="font-semibold">Loaded {uploadState.fileName}</p>
              <span className="text-sm text-emerald-200/80">
                {uploadState.rowCount} data rows detected
              </span>
            </div>
            <div>
              <p className="text-sm text-emerald-200/70 mb-2">
                Preview (first rows):
              </p>
              <pre className="bg-black/40 text-white/90 text-xs rounded-md p-3 overflow-x-auto whitespace-pre-wrap">
                {uploadState.preview || "No preview available."}
              </pre>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default DataPortal;
