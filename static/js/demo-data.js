/**
 * Pre-computed demo data for the SelectAnyTree click explorer.
 *
 * HOW TO POPULATE:
 * 1. Run the webdemo on a GPU server for each scene.
 * 2. Screenshot the base scene (no segmentation) → save as static/images/demo_<scene>_base.png
 * 3. For each tree hotspot, screenshot with that tree segmented → demo_<scene>_tree<N>.png
 * 4. Update the hotspots array with (x%, y%) positions as percentages of image dimensions.
 *
 * Positions are given as percentages [0–100] of the rendered image width/height.
 */

window.DEMO_SCENES = {
  nibio: {
    label: "Norway (NIBIO)",
    base: "static/images/demo_nibio_base.png",
    hotspots: [
      // { x: 35, y: 28, result: "static/images/demo_nibio_tree1.png", iou: 84.3, pts: 1820, id: "Tree #1" },
      // { x: 55, y: 45, result: "static/images/demo_nibio_tree2.png", iou: 79.1, pts: 2240, id: "Tree #2" },
      // { x: 70, y: 30, result: "static/images/demo_nibio_tree3.png", iou: 81.7, pts: 1650, id: "Tree #3" },
      // Uncomment and fill in actual positions once images are ready.
      // Placeholder: three evenly-spaced hotspots for layout preview.
      { x: 30, y: 35, result: "", iou: "—", pts: "—", id: "Tree #1 (placeholder)" },
      { x: 55, y: 50, result: "", iou: "—", pts: "—", id: "Tree #2 (placeholder)" },
      { x: 72, y: 28, result: "", iou: "—", pts: "—", id: "Tree #3 (placeholder)" },
    ]
  },

  bluecat: {
    label: "Australia (BlueCat)",
    base: "static/images/demo_bluecat_base.png",
    hotspots: [
      { x: 25, y: 40, result: "", iou: "—", pts: "—", id: "Tree #1 (placeholder)" },
      { x: 50, y: 55, result: "", iou: "—", pts: "—", id: "Tree #2 (placeholder)" },
      { x: 68, y: 35, result: "", iou: "—", pts: "—", id: "Tree #3 (placeholder)" },
    ]
  },

  culs: {
    label: "Czech Rep. (CULS)",
    base: "static/images/demo_culs_base.png",
    hotspots: [
      { x: 40, y: 30, result: "", iou: "—", pts: "—", id: "Tree #1 (placeholder)" },
      { x: 60, y: 48, result: "", iou: "—", pts: "—", id: "Tree #2 (placeholder)" },
      { x: 75, y: 60, result: "", iou: "—", pts: "—", id: "Tree #3 (placeholder)" },
    ]
  }
};
