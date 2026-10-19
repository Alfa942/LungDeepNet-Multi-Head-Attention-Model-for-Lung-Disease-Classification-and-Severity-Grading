import React, { useEffect, useRef } from 'react';

export type ColormapType = 'jet' | 'turbo' | 'viridis' | 'plasma' | 'inferno';

interface HeatmapCanvasProps {
  imageSrc: string;
  heatmapGrid: number[][]; // e.g. 7x7 normalized [0, 1]
  alpha: number; // 0 to 1
  colormap: ColormapType;
  threshold: number; // 0 to 1
  showBBox?: boolean;
  bBox?: { x: number; y: number; width: number; height: number; label: string };
  width?: number;
  height?: number;
  className?: string;
  id?: string;
}

// Colormap lookup functions
function getColormapRgb(val: number, cmap: ColormapType): [number, number, number] {
  const v = Math.min(1, Math.max(0, val));
  
  if (cmap === 'jet') {
    // Jet: blue -> cyan -> yellow -> red
    const r = Math.min(1, Math.max(0, 1.5 - Math.abs(v * 4 - 3)));
    const g = Math.min(1, Math.max(0, 1.5 - Math.abs(v * 4 - 2)));
    const b = Math.min(1, Math.max(0, 1.5 - Math.abs(v * 4 - 1)));
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  } else if (cmap === 'turbo') {
    // Turbo smooth rainbow
    const r = Math.min(255, Math.max(0, Math.round(34.61 + v * (1172.6 + v * (-4949.4 + v * (7514.8 - v * 3583.5))))));
    const g = Math.min(255, Math.max(0, Math.round(23.31 + v * (557.3 + v * (122.5 + v * (-2694.0 + v * 1999.0))))));
    const b = Math.min(255, Math.max(0, Math.round(27.2 + v * (3211.1 + v * (-15327.9 + v * (27814.0 - v * 16328.0))))));
    return [r, g, b];
  } else if (cmap === 'viridis') {
    // Viridis: purple -> teal -> yellow
    const r = Math.round(255 * (0.28 + 0.72 * Math.pow(v, 2.5)));
    const g = Math.round(255 * (0.13 + 0.82 * Math.sin(v * Math.PI * 0.9)));
    const b = Math.round(255 * (0.45 + 0.55 * Math.cos(v * Math.PI * 0.75)));
    return [r, g, b];
  } else if (cmap === 'plasma') {
    // Plasma: purple -> magenta -> orange -> yellow
    const r = Math.round(255 * (0.05 + 0.95 * Math.pow(v, 0.7)));
    const g = Math.round(255 * (0.02 + 0.95 * Math.pow(v, 1.8)));
    const b = Math.round(255 * (0.55 * (1 - v) + 0.2 * Math.sin(v * Math.PI)));
    return [r, g, b];
  } else {
    // Inferno: black -> red -> gold -> yellow
    const r = Math.round(255 * Math.min(1, Math.pow(v, 0.6) * 1.2));
    const g = Math.round(255 * Math.max(0, Math.pow(v, 1.8) * 0.95));
    const b = Math.round(255 * Math.max(0, Math.sin(v * Math.PI * 0.8) * 0.3 + (v > 0.8 ? (v - 0.8) * 4 : 0)));
    return [r, g, b];
  }
}

// Bilinear interpolation for 2D grid
function bilinearSample(grid: number[][], u: number, v: number): number {
  const h = grid.length;
  const w = grid[0].length;
  
  const gx = u * (w - 1);
  const gy = v * (h - 1);
  
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  
  const tx = gx - x0;
  const ty = gy - y0;
  
  const v00 = grid[y0][x0];
  const v10 = grid[y0][x1];
  const v01 = grid[y1][x0];
  const v11 = grid[y1][x1];
  
  const top = v00 * (1 - tx) + v10 * tx;
  const bottom = v01 * (1 - tx) + v11 * tx;
  
  return top * (1 - ty) + bottom * ty;
}

export const HeatmapCanvas: React.FC<HeatmapCanvasProps> = ({
  imageSrc,
  heatmapGrid,
  alpha,
  colormap,
  threshold,
  showBBox = false,
  bBox,
  width = 440,
  height = 440,
  className = '',
  id = 'heatmap-canvas',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageSrc;

    img.onload = () => {
      // 1. Draw base chest X-ray image
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);

      // 2. Generate and overlay interpolated heatmap
      if (alpha > 0.01 && heatmapGrid && heatmapGrid.length > 0) {
        const heatmapImgData = ctx.createImageData(width, height);
        const data = heatmapImgData.data;

        for (let y = 0; y < height; y++) {
          const v = y / (height - 1);
          for (let x = 0; x < width; x++) {
            const u = x / (width - 1);
            let val = bilinearSample(heatmapGrid, u, v);

            // Apply threshold cutoff
            if (val < threshold) {
              val = 0;
            } else {
              // Rescale to 0..1 above threshold
              val = (val - threshold) / (1 - threshold + 1e-6);
            }

            const pixelIdx = (y * width + x) * 4;

            if (val > 0.02) {
              const [r, g, b] = getColormapRgb(val, colormap);
              data[pixelIdx] = r;
              data[pixelIdx + 1] = g;
              data[pixelIdx + 2] = b;
              data[pixelIdx + 3] = Math.round(val * alpha * 255);
            } else {
              data[pixelIdx + 3] = 0; // Transparent
            }
          }
        }

        // Draw blended heatmap overlay using temporary offscreen canvas for crisp blending
        const offscreen = document.createElement('canvas');
        offscreen.width = width;
        offscreen.height = height;
        const offCtx = offscreen.getContext('2d');
        if (offCtx) {
          offCtx.putImageData(heatmapImgData, 0, 0);
          ctx.drawImage(offscreen, 0, 0);
        }
      }

      // 3. Draw Ground Truth Bounding Box / Annotations if enabled
      if (showBBox && bBox) {
        const scaleX = width / 480;
        const scaleY = height / 480;
        const bx = bBox.x * scaleX;
        const by = bBox.y * scaleY;
        const bw = bBox.width * scaleX;
        const bh = bBox.height * scaleY;

        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.setLineDash([]);

        // Label pill
        ctx.fillStyle = 'rgba(6, 182, 212, 0.9)';
        ctx.fillRect(bx, by - 22, ctx.measureText(bBox.label).width + 16, 20);
        ctx.fillStyle = '#020617';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(bBox.label, bx + 6, by - 8);
      }
    };
  }, [imageSrc, heatmapGrid, alpha, colormap, threshold, showBBox, bBox, width, height]);

  return (
    <canvas
      ref={canvasRef}
      id={id}
      width={width}
      height={height}
      className={`rounded-xl shadow-2xl border border-slate-800 bg-slate-950 ${className}`}
    />
  );
};
