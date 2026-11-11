import React, { useEffect, useState, useRef } from "react";
import dynamic from "next/dynamic";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
});

export default function GraphifyView({ telemetry }) {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 350 });

  useEffect(() => {
    if (!telemetry || telemetry.length === 0) return;

    // Drastically reduced the 'val' size so they aren't giant blobs!
    const nodes = [{ id: "User Session", group: 0, val: 2, color: "#0f172a" }];
    const links = [];

    telemetry.forEach((plant) => {
      nodes.push({ id: plant.species, group: 1, val: 1.5, color: "#10b981" });
      links.push({
        source: "User Session",
        target: plant.species,
        name: "detected",
      });
    });

    setGraphData({ nodes, links });
  }, [telemetry]);

  // Make the WebGL canvas perfectly responsive to your UI panel
  useEffect(() => {
    if (containerRef.current) {
      setDimensions({
        width: containerRef.current.offsetWidth,
        height: 350,
      });
    }
    
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({ width: containerRef.current.offsetWidth, height: 350 });
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  if (!telemetry || telemetry.length === 0) return null;

  return (
    <div
      ref={containerRef}
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
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeRelSize={6}
        linkColor={() => "#cbd5e1"}
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={0.005}
        d3VelocityDecay={0.2}
        // Custom Canvas drawing to put the text directly on the nodes!
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.id;
          const fontSize = 16 / globalScale;
          ctx.font = `bold ${fontSize}px "VT323", monospace`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.4);

          ctx.fillStyle = "rgba(248, 250, 252, 0.9)";
          ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);

          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = node.color;
          ctx.fillText(label, node.x, node.y);
          node.__bckgDimensions = bckgDimensions;
        }}
        nodePointerAreaPaint={(node, color, ctx) => {
          ctx.fillStyle = color;
          const bckgDimensions = node.__bckgDimensions;
          bckgDimensions && ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);
        }}
      />
    </div>
  );
}