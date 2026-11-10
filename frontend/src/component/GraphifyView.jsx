import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";

// Dynamically import the graph to prevent Next.js SSR window errors
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
});

export default function GraphifyView({ telemetry }) {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

  useEffect(() => {
    if (!telemetry || telemetry.length === 0) return;

    // 1. Create the central User Session hub
    const nodes = [{ id: "User Session", group: 0, val: 20, color: "#0f172a" }];
    const links = [];

    // 2. Loop through visual telemetry and link discovered plants
    telemetry.forEach((plant) => {
      nodes.push({ id: plant.species, group: 1, val: 12, color: "#10b981" });
      links.push({
        source: "User Session",
        target: plant.species,
        name: "detected",
      });
    });

    setGraphData({ nodes, links });
  }, [telemetry]);

  if (!telemetry || telemetry.length === 0) return null;

  return (
    <div
      style={{
        height: "350px",
        width: "100%",
        border: "4px solid #a7f3d0",
        backgroundColor: "#f8fafc",
        marginTop: "20px",
        position: "relative",
      }}
    >
      <h4
        style={{
          margin: 0,
          padding: "8px 12px",
          backgroundColor: "#a7f3d0",
          color: "#064e3b",
          fontFamily: "'VT323', monospace",
          fontSize: "20px",
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
        }}
      >
        🕸️ Graphify Knowledge Web
      </h4>
      <ForceGraph2D
        graphData={graphData}
        nodeLabel="id"
        nodeColor="color"
        nodeRelSize={6}
        linkColor={() => "#94a3b8"}
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={0.005}
        d3VelocityDecay={0.3}
      />
    </div>
  );
}
