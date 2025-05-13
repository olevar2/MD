import React, { useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

interface Node {
  id: string;
  label: string;
}

interface Edge {
  source: string;
  target: string;
}

interface NetworkGraphProps {
  graph: {
    nodes: Node[];
    edges: Edge[];
  };
}

const NetworkGraph: React.FC<NetworkGraphProps> = ({ graph }) => {
  const graphRef = useRef<any>();

  // Convert the graph data to the format expected by ForceGraph2D
  const graphData = {
    nodes: graph.nodes.map(node => ({
      id: node.id,
      label: node.label,
      // Add visual properties
      color: node.id.includes('close') ? '#ff6b6b' : '#4ecdc4'
    })),
    links: graph.edges.map(edge => ({
      source: edge.source,
      target: edge.target,
      // Add visual properties
      color: '#718096',
      width: 2
    }))
  };

  useEffect(() => {
    // When the component mounts, adjust the graph visualization
    if (graphRef.current) {
      // Zoom to fit the graph
      graphRef.current.zoomToFit(400);
    }
  }, [graph]);

  return (
    <ForceGraph2D
      ref={graphRef}
      graphData={graphData}
      nodeLabel="label"
      nodeRelSize={6}
      linkDirectionalArrowLength={6}
      linkDirectionalArrowRelPos={1}
      linkCurvature={0.25}
      nodeCanvasObject={(node: any, ctx, globalScale) => {
        const label = node.label;
        const fontSize = 12/globalScale;
        ctx.font = `${fontSize}px Sans-Serif`;
        const textWidth = ctx.measureText(label).width;
        const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

        // Node background
        ctx.fillStyle = node.color;
        ctx.fillRect(
          node.x - bckgDimensions[0] / 2, 
          node.y - bckgDimensions[1] / 2, 
          bckgDimensions[0], 
          bckgDimensions[1]
        );

        // Node text
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, node.x, node.y);

        // Return shape that's being hovered (for nodes hovering detection)
        return {
          x: node.x,
          y: node.y,
          width: bckgDimensions[0],
          height: bckgDimensions[1]
        };
      }}
    />
  );
};

export default NetworkGraph;
