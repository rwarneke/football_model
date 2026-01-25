#!/usr/bin/env python3
"""
Extract dominant colors from flag images and generate a TypeScript mapping.
Uses color clustering to find distinct main colors rather than similar shades.
"""
import os
from pathlib import Path
from PIL import Image
import json
from collections import Counter
import math

def color_distance(rgb1, rgb2):
    """
    Calculate perceptual color distance using weighted Euclidean distance.
    This better matches human perception than simple RGB distance.
    """
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    
    # Weighted Euclidean distance (weights approximate human color perception)
    # Red and green are more perceptually significant than blue
    r_mean = (r1 + r2) / 2
    delta_r = r1 - r2
    delta_g = g1 - g2
    delta_b = b1 - b2
    
    # Use a simpler but effective distance metric
    # Weight red and green more heavily
    distance = math.sqrt(
        2 * (delta_r ** 2) +
        4 * (delta_g ** 2) +
        3 * (delta_b ** 2)
    )
    
    return distance

def merge_closest_colors(color_counts, merge_threshold=60):
    """
    Iteratively merge the closest pair of colors until the minimum distance
    between any two colors is >= merge_threshold.
    Uses a priority queue with lazy invalidation for efficiency.
    """
    import heapq

    # Start with each unique color as its own cluster centroid
    clusters = {}
    for idx, ((r, g, b), count) in enumerate(color_counts.items()):
        clusters[idx] = {
            "color": (r, g, b),
            "count": count,
            "version": 0,
            "active": True,
        }

    active_ids = set(clusters.keys())
    if len(active_ids) <= 1:
        return [clusters[i] for i in active_ids]

    # Build initial heap of all pair distances
    heap = []
    ids = list(active_ids)
    for i in range(len(ids)):
        ci = clusters[ids[i]]["color"]
        for j in range(i + 1, len(ids)):
            cj = clusters[ids[j]]["color"]
            dist = color_distance(ci, cj)
            heapq.heappush(
                heap,
                (dist, ids[i], ids[j], clusters[ids[i]]["version"], clusters[ids[j]]["version"]),
            )

    next_id = max(active_ids) + 1

    while heap:
        dist, a, b, ver_a, ver_b = heapq.heappop(heap)
        ca = clusters.get(a)
        cb = clusters.get(b)

        if (
            ca is None
            or cb is None
            or not ca["active"]
            or not cb["active"]
            or ca["version"] != ver_a
            or cb["version"] != ver_b
        ):
            continue

        if dist >= merge_threshold:
            break

        total = ca["count"] + cb["count"]
        r_val = round((ca["color"][0] * ca["count"] + cb["color"][0] * cb["count"]) / total)
        g_val = round((ca["color"][1] * ca["count"] + cb["color"][1] * cb["count"]) / total)
        b_val = round((ca["color"][2] * ca["count"] + cb["color"][2] * cb["count"]) / total)

        ca["active"] = False
        cb["active"] = False
        active_ids.discard(a)
        active_ids.discard(b)

        clusters[next_id] = {
            "color": (r_val, g_val, b_val),
            "count": total,
            "version": 0,
            "active": True,
        }
        new_id = next_id
        next_id += 1
        active_ids.add(new_id)

        # Push distances from new cluster to all active clusters
        new_color = clusters[new_id]["color"]
        for other_id in active_ids:
            if other_id == new_id:
                continue
            other = clusters[other_id]
            dist = color_distance(new_color, other["color"])
            heapq.heappush(
                heap,
                (dist, new_id, other_id, clusters[new_id]["version"], other["version"]),
            )

    return [clusters[i] for i in active_ids]

def is_very_light_color(r, g, b, threshold=240):
    """
    Check if a color is very light (close to white).
    """
    return r > threshold and g > threshold and b > threshold

def is_very_dark_color(r, g, b, threshold=15):
    """
    Check if a color is very dark (close to black).
    """
    return r < threshold and g < threshold and b < threshold

def get_dominant_colors(
    image_path,
    similarity_threshold=220,
    sample_size=120,
):
    """
    Extract colors from an image by merging the closest colors until the
    minimum distance between any two colors meets the threshold.
    The threshold is the only tuning parameter.
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for faster processing while maintaining aspect ratio
        img.thumbnail((sample_size, sample_size), Image.Resampling.LANCZOS)
        
        # Get all pixels (pre-quantize to reduce unique colors for speed)
        pixels = []
        for (r, g, b) in img.getdata():
            qr = (r // 4) * 4
            qg = (g // 4) * 4
            qb = (b // 4) * 4
            pixels.append((qr, qg, qb))
        
        # Count color frequencies
        color_counts = Counter(pixels)
        
        clusters = merge_closest_colors(color_counts, similarity_threshold)
        clusters.sort(key=lambda x: x["count"], reverse=True)
        selected_colors = [c["color"] for c in clusters]

        # Convert to hex
        colors = []
        for (r, g, b) in selected_colors:
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            colors.append(hex_color)
        
        # If we don't have enough distinct colors, we're done
        # (some flags genuinely have fewer colors)
        return colors if colors else ["#FF0000", "#0000FF", "#FFFFFF"]
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ["#FF0000", "#0000FF", "#FFFFFF"]

def main():
    flags_dir = Path("reference_data/flags")
    output_file = Path("web/lib/flag-colors.ts")
    
    if not flags_dir.exists():
        print(f"Flags directory not found: {flags_dir}")
        return
    
    team_colors = {}
    
    # Process each flag file
    for flag_file in sorted(flags_dir.glob("*.png")):
        team_name = flag_file.stem  # filename without extension
        colors = get_dominant_colors(flag_file)
        team_colors[team_name] = colors
        print(f"{team_name}: {colors}")
    
    # Generate TypeScript file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("// Auto-generated flag colors mapping\n")
        f.write("// Generated by misc/extract_flag_colors.py\n\n")
        f.write("export const FLAG_COLORS: Record<string, string[]> = {\n")
        for team, colors in sorted(team_colors.items()):
            colors_str = ", ".join([f'"{c}"' for c in colors])
            f.write(f'  "{team}": [{colors_str}],\n')
        f.write("};\n")
    
    print(f"\nGenerated {output_file} with {len(team_colors)} teams")

if __name__ == "__main__":
    main()
