import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Network } from "lucide-react";

interface Node {
  id: string;
  name: string;
  type: string;
}

interface Edge {
  source: string;
  target: string;
  relation: string;
}

interface ConceptMapProps {
  nodes: Node[];
  edges: Edge[];
  stats?: {
    papers_indexed: number;
    concepts_extracted: number;
    relations_found: number;
  };
}

export function ConceptMap({ nodes, edges, stats }: ConceptMapProps) {
  // Simple layout: circular
  const layout = useMemo(() => {
    const width = 600;
    const height = 400;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 40;

    const nodePositions = nodes.map((node, i) => {
      const angle = (i / nodes.length) * 2 * Math.PI;
      return {
        ...node,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      };
    });

    const nodeMap = new Map(nodePositions.map(n => [n.id, n]));

    return { nodePositions, nodeMap, width, height };
  }, [nodes]);

  if (!nodes || nodes.length === 0) {
    return (
      <Card className="h-full flex items-center justify-center min-h-[300px]">
        <div className="text-center text-muted-foreground">
          <Network className="w-12 h-12 mx-auto mb-2 opacity-20" />
          <p>No concept map data available yet.</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2 border-b">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm uppercase tracking-wider flex items-center gap-2">
            <Network className="w-4 h-4 text-primary" />
            Global Concept Map
          </CardTitle>
          {stats && (
            <div className="flex gap-3 text-[10px] text-muted-foreground font-mono">
              <span>{stats.papers_indexed} PAPERS</span>
              <span>{stats.concepts_extracted} CONCEPTS</span>
              <span>{stats.relations_found} RELATIONS</span>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-0 relative overflow-hidden bg-slate-50/50 dark:bg-slate-950/50">
        <svg width="100%" height="100%" viewBox={`0 0 ${layout.width} ${layout.height}`} className="w-full h-full">
          <defs>
            <marker
              id="arrowhead"
              markerWidth="6"
              markerHeight="4"
              refX="16"
              refY="2"
              orient="auto"
              className="fill-muted-foreground/40"
            >
              <polygon points="0 0, 6 2, 0 4" />
            </marker>
          </defs>
          <g>
            {edges.map((edge, i) => {
              const source = layout.nodeMap.get(edge.source);
              const target = layout.nodeMap.get(edge.target);
              if (!source || !target) return null;
              return (
                <line
                  key={i}
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  stroke="currentColor"
                  strokeWidth="1"
                  className="text-border"
                  markerEnd="url(#arrowhead)"
                />
              );
            })}
          </g>
          <g>
            {layout.nodePositions.map((node) => (
              <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                <circle r="4" className="fill-background stroke-primary stroke-2" />
                <text
                  dy="-8"
                  textAnchor="middle"
                  className="text-[8px] fill-foreground font-medium pointer-events-none select-none"
                >
                  {node.name}
                </text>
              </g>
            ))}
          </g>
        </svg>
      </CardContent>
    </Card>
  );
}
