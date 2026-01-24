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

def group_similar_colors(color_counts, threshold=40):
    """
    Group colors that are perceptually similar together.
    Returns a list of color groups, where each group contains similar colors.
    Uses a more aggressive threshold for very light or very dark colors.
    """
    # Sort colors by frequency (most common first)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    groups = []
    used_colors = set()
    
    for (r, g, b), count in sorted_colors:
        # Skip if this color is already in a group
        if (r, g, b) in used_colors:
            continue
        
        # Start a new group with this color
        current_group = [((r, g, b), count)]
        used_colors.add((r, g, b))
        
        # Use more aggressive threshold for very light or very dark colors
        is_light = is_very_light_color(r, g, b)
        is_dark = is_very_dark_color(r, g, b)
        current_threshold = threshold * 1.5 if (is_light or is_dark) else threshold
        
        # Find all similar colors
        for (r2, g2, b2), count2 in sorted_colors:
            if (r2, g2, b2) in used_colors:
                continue
            
            # Check if colors are similar
            dist = color_distance((r, g, b), (r2, g2, b2))
            if dist < current_threshold:
                current_group.append(((r2, g2, b2), count2))
                used_colors.add((r2, g2, b2))
        
        groups.append(current_group)
    
    return groups

def get_representative_color(group):
    """
    Get the most frequent color from a group of similar colors.
    """
    # Return the color with the highest frequency
    return max(group, key=lambda x: x[1])[0]

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

def get_dominant_colors(image_path, max_colors=3, similarity_threshold=40):
    """
    Extract up to max_colors distinct dominant colors from an image.
    Uses color clustering to avoid selecting similar shades.
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for faster processing while maintaining aspect ratio
        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
        
        # Get all pixels
        pixels = list(img.getdata())
        
        # Count color frequencies
        color_counts = Counter(pixels)
        
        # Group similar colors together
        color_groups = group_similar_colors(color_counts, similarity_threshold)
        
        # Get representative color from each group, sorted by total frequency
        # (sum of frequencies of all colors in the group)
        group_representatives = []
        for group in color_groups:
            total_freq = sum(count for _, count in group)
            rep_color = get_representative_color(group)
            group_representatives.append((rep_color, total_freq))
        
        # Sort by total frequency (most dominant groups first)
        group_representatives.sort(key=lambda x: x[1], reverse=True)
        
        # Select distinct colors, ensuring minimum distance between them
        selected_colors = []
        selected_frequencies = []
        min_color_distance = 50  # Minimum distance between selected colors
        
        for (r, g, b), freq in group_representatives:
            # Check if this color is sufficiently different from already selected colors
            is_distinct = True
            for (r2, g2, b2) in selected_colors:
                dist = color_distance((r, g, b), (r2, g2, b2))
                if dist < min_color_distance:
                    is_distinct = False
                    break
            
            if is_distinct:
                selected_colors.append((r, g, b))
                selected_frequencies.append(freq)
                
                if len(selected_colors) >= max_colors:
                    break
        
        # Post-process: If we have 3 colors, check if the third is significantly less frequent
        # Only drop it if it's truly insignificant (less than 10% of total)
        if len(selected_colors) == 3:
            total_freq = sum(selected_frequencies)
            if total_freq > 0:
                third_ratio = selected_frequencies[2] / total_freq
                # Only drop third color if it represents less than 10% of pixels
                # and the first two colors are both significant
                if third_ratio < 0.10 and selected_frequencies[0] > 0 and selected_frequencies[1] > 0:
                    first_ratio = selected_frequencies[0] / total_freq
                    second_ratio = selected_frequencies[1] / total_freq
                    # Only drop if first two are both substantial (at least 30% each)
                    if first_ratio >= 0.30 and second_ratio >= 0.30:
                        selected_colors = selected_colors[:2]
        
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
        colors = get_dominant_colors(flag_file, max_colors=3)
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
