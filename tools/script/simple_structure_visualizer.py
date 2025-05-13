#!/usr/bin/env python3
"""
Enhanced Project Structure Visualizer

This script analyzes the project structure and creates:
1. A JSON file with the complete directory structure
2. A colorful visualization of the project structure with circles and arrows
"""

import os
import json
import argparse
import random
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from PIL import Image, ImageDraw, ImageFont

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    ".git", ".github", ".pytest_cache", "__pycache__",
    "node_modules", ".venv", "venv", "env", ".vscode"
}

# File extensions to include in analysis
INCLUDE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yml", ".yaml",
    ".md", ".sql", ".html", ".css", ".scss", ".sh", ".bat", ".ps1",
    ".dockerfile", ".dockerignore", ".gitignore", ".env", ".toml"
}

# Colors
BG_COLOR = (255, 255, 255)  # White background
TEXT_COLOR = (0, 0, 0)  # Black text
LINE_COLOR = (100, 100, 100)  # Darker gray lines for better visibility
ARROW_COLOR = (50, 50, 50)  # Dark gray arrows

# Vibrant color palette for directories and files
DIRECTORY_COLORS = [
    (255, 87, 51),   # Vibrant Red
    (51, 255, 87),   # Vibrant Green
    (51, 87, 255),   # Vibrant Blue
    (255, 195, 0),   # Vibrant Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (0, 128, 255),   # Sky Blue
    (255, 0, 128),   # Pink
]

FILE_COLORS = [
    (173, 216, 230),  # Light Blue
    (144, 238, 144),  # Light Green
    (255, 182, 193),  # Light Pink
    (255, 218, 185),  # Peach
    (221, 160, 221),  # Plum
    (176, 224, 230),  # Powder Blue
    (255, 250, 205),  # Lemon Chiffon
    (240, 230, 140),  # Khaki
    (255, 228, 196),  # Bisque
    (255, 222, 173),  # Navajo White
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create simple project structure visualization")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth to visualize"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3000,
        help="Width of the image in pixels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2000,
        help="Height of the image in pixels"
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        default=list(EXCLUDE_DIRS),
        help="Directories to exclude from analysis"
    )
    return parser.parse_args()

def analyze_directory(
    directory: Path,
    max_depth: int = 5,
    current_depth: int = 0,
    exclude_dirs: Set[str] = EXCLUDE_DIRS
) -> Dict[str, Any]:
    """
    Recursively analyze a directory structure.

    Args:
        directory: Directory to analyze
        max_depth: Maximum depth to analyze
        current_depth: Current depth in the recursion
        exclude_dirs: Directories to exclude from analysis

    Returns:
        Dictionary representing the directory structure
    """
    if current_depth > max_depth:
        return {"name": directory.name, "type": "directory", "truncated": True}

    result = {
        "name": directory.name,
        "type": "directory",
        "path": str(directory),
        "children": []
    }

    try:
        items = list(directory.iterdir())

        # Process directories first
        dirs = [item for item in items if item.is_dir() and item.name not in exclude_dirs]
        for dir_path in sorted(dirs):
            child = analyze_directory(
                dir_path,
                max_depth,
                current_depth + 1,
                exclude_dirs
            )
            result["children"].append(child)

        # Then process files
        files = [item for item in items if item.is_file() and item.suffix.lower() in INCLUDE_EXTENSIONS]
        for file_path in sorted(files):
            result["children"].append({
                "name": file_path.name,
                "type": "file",
                "path": str(file_path),
                "extension": file_path.suffix.lower()
            })

        # Add counts
        result["file_count"] = len(files)
        result["dir_count"] = len(dirs)
        result["total_count"] = len(files) + len(dirs)

    except PermissionError:
        result["error"] = "Permission denied"
    except Exception as e:
        result["error"] = str(e)

    return result

def count_nodes(node, max_depth, current_depth=0):
    """Count the number of visible nodes."""
    if current_depth > max_depth:
        return 1

    count = 1  # Count the current node
    if node["type"] == "directory" and "children" in node and current_depth < max_depth:
        for child in node["children"]:
            count += count_nodes(child, max_depth, current_depth + 1)

    return count

def draw_tree(draw, font, node, x, y, level_width, node_height, max_depth, current_depth=0):
    """
    Draw the tree structure.

    Args:
        draw: PIL ImageDraw object
        font: PIL ImageFont object
        node: Current node to draw
        x, y: Position to draw the node
        level_width: Width between levels
        node_height: Height of each node
        max_depth: Maximum depth to visualize
        current_depth: Current depth in the recursion

    Returns:
        y_offset: The y position after drawing this node and its children
    """
    if current_depth > max_depth:
        return y

    # Determine color based on node type
    if node["type"] == "directory":
        # Choose a color based on depth
        color_idx = current_depth % len(DIRECTORY_COLORS)
        color = DIRECTORY_COLORS[color_idx]
    else:
        # Choose a color based on file extension
        ext = os.path.splitext(node["name"])[1].lower()
        color_idx = hash(ext) % len(FILE_COLORS)
        color = FILE_COLORS[color_idx]

    # Prepare label
    if node["type"] == "directory":
        file_count = node.get("file_count", 0)
        dir_count = node.get("dir_count", 0)
        label = f"{node['name']} ({file_count} files, {dir_count} dirs)"
    else:
        label = node["name"]

    # Draw the node rectangle
    rect_width = 300
    draw.rectangle([(x, y), (x + rect_width, y + node_height)], fill=color, outline=(0, 0, 0))

    # Draw the text
    text_x = x + 10
    text_y = y + (node_height - font.getbbox(label)[3]) // 2
    draw.text((text_x, text_y), label, fill=TEXT_COLOR, font=font)

    # Process children
    if node["type"] == "directory" and "children" in node and current_depth < max_depth:
        # Sort children: directories first, then files
        children = sorted(node["children"], key=lambda x: (x["type"] != "directory", x["name"]))

        # Limit the number of children to display
        max_children = 20
        if len(children) > max_children:
            visible_children = children[:max_children]
            has_more = True
        else:
            visible_children = children
            has_more = False

        # Calculate starting y position for children
        child_y = y

        # Draw children
        for child in visible_children:
            # Draw connecting line
            mid_y = child_y + node_height // 2
            draw.line([(x + rect_width, y + node_height // 2), (x + rect_width + 20, mid_y), (x + level_width, mid_y)],
                      fill=LINE_COLOR, width=2)

            # Draw the child
            child_y = draw_tree(draw, font, child, x + level_width, child_y, level_width, node_height,
                               max_depth, current_depth + 1)

            # Add spacing between siblings
            child_y += node_height // 2

        # If there are more children than we're showing
        if has_more:
            # Draw connecting line
            mid_y = child_y + node_height // 2
            draw.line([(x + rect_width, y + node_height // 2), (x + rect_width + 20, mid_y), (x + level_width, mid_y)],
                      fill=LINE_COLOR, width=2)

            # Draw "more" indicator
            draw.rectangle([(x + level_width, child_y), (x + level_width + rect_width, child_y + node_height)],
                          fill=(220, 220, 220), outline=(0, 0, 0))
            more_text = f"... {len(children) - max_children} more items"
            draw.text((x + level_width + 10, child_y + (node_height - font.getbbox(more_text)[3]) // 2),
                     more_text, fill=TEXT_COLOR, font=font)

            child_y += node_height + node_height // 2

        return max(child_y, y + node_height)
    else:
        return y + node_height

def create_visualization(structure, output_path, max_depth=2, width=3000, height=2000):
    """
    Create a colorful visualization of the project structure with circles and arrows.

    Args:
        structure: Dictionary representing the directory structure
        output_path: Path to save the visualization
        max_depth: Maximum depth to visualize
        width, height: Dimensions of the image
    """
    # Create a new image
    img = Image.new('RGB', (width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()
        title_font = font

    # Draw the title
    title = f"Project Structure: {structure['name']}"
    draw.text((20, 20), title, fill=TEXT_COLOR, font=title_font)

    # Create a mapping of nodes to positions
    nodes = []
    connections = []

    # Extract nodes and connections
    def extract_nodes(node, parent=None, depth=0, path=""):
    """
    Extract nodes.
    
    Args:
        node: Description of node
        parent: Description of parent
        depth: Description of depth
        path: Description of path
    
    """

        if depth > max_depth:
            return

        # Create a unique ID for this node
        node_id = len(nodes)

        # Add this node
        nodes.append({
            "id": node_id,
            "name": node["name"],
            "type": node["type"],
            "depth": depth,
            "file_count": node.get("file_count", 0),
            "dir_count": node.get("dir_count", 0),
            "path": path + "/" + node["name"]
        })

        # Add connection to parent
        if parent is not None:
            connections.append((parent, node_id))

        # Process children
        if node["type"] == "directory" and "children" in node and depth < max_depth:
            # Sort children: directories first, then files
            children = sorted(node["children"], key=lambda x: (x["type"] != "directory", x["name"]))

            # Limit the number of children to display
            max_children = 15
            if len(children) > max_children:
                visible_children = children[:max_children]
            else:
                visible_children = children

            # Process children
            for child in visible_children:
                extract_nodes(child, node_id, depth + 1, path + "/" + node["name"])

    # Start extraction from root
    extract_nodes(structure, None, 0, "")

    # Calculate positions using a force-directed layout
    positions = calculate_force_directed_layout(nodes, connections, width, height)

    # Draw connections (arrows)
    for source, target in connections:
        draw_arrow(draw, positions[source], positions[target])

    # Draw nodes (circles)
    for i, node in enumerate(nodes):
        draw_node(draw, font, node, positions[i])

    # Save the image
    img.save(output_path)
    print(f"Enhanced visualization saved to {output_path}")

def calculate_force_directed_layout(nodes, connections, width, height, iterations=100):
    """
    Calculate node positions using a force-directed layout algorithm.

    Args:
        nodes: List of nodes
        connections: List of connections (source, target)
        width, height: Dimensions of the image
        iterations: Number of iterations for the layout algorithm

    Returns:
        List of (x, y) positions for each node
    """
    # Initialize positions randomly
    positions = []
    margin = 100  # Margin from the edges
    for node in nodes:
        depth = node["depth"]
        # Position nodes in layers based on depth
        x = margin + (width - 2 * margin) * (depth / max(1, max(n["depth"] for n in nodes)))
        # Add some randomness to y
        y = margin + random.random() * (height - 2 * margin)
        positions.append((x, y))

    # Define forces
    def repulsive_force(pos1, pos2, strength=5000):
    """
    Repulsive force.
    
    Args:
        pos1: Description of pos1
        pos2: Description of pos2
        strength: Description of strength
    
    """

        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        distance = max(1, math.sqrt(dx * dx + dy * dy))
        force = strength / (distance * distance)
        return force * dx / distance, force * dy / distance

    def attractive_force(pos1, pos2, strength=0.2):
    """
    Attractive force.
    
    Args:
        pos1: Description of pos1
        pos2: Description of pos2
        strength: Description of strength
    
    """

        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        distance = max(1, math.sqrt(dx * dx + dy * dy))
        force = strength * distance
        return -force * dx / distance, -force * dy / distance

    # Run force-directed layout
    for _ in range(iterations):
        # Calculate forces
        forces = [(0, 0) for _ in range(len(nodes))]

        # Repulsive forces between all nodes
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    fx, fy = repulsive_force(positions[i], positions[j])
                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)

        # Attractive forces along connections
        for source, target in connections:
            fx, fy = attractive_force(positions[source], positions[target])
            forces[source] = (forces[source][0] + fx, forces[source][1] + fy)
            forces[target] = (forces[target][0] - fx, forces[target][1] - fy)

        # Apply forces
        for i in range(len(nodes)):
            x, y = positions[i]
            fx, fy = forces[i]
            # Limit force magnitude
            force_mag = math.sqrt(fx * fx + fy * fy)
            if force_mag > 10:
                fx = fx * 10 / force_mag
                fy = fy * 10 / force_mag
            # Update position
            x += fx
            y += fy
            # Keep within bounds
            x = max(margin, min(width - margin, x))
            y = max(margin, min(height - margin, y))
            positions[i] = (x, y)

    return positions

def draw_arrow(draw, start, end, color=ARROW_COLOR, width=2):
    """
    Draw an arrow from start to end.

    Args:
        draw: PIL ImageDraw object
        start: Start position (x, y)
        end: End position (x, y)
        color: Arrow color
        width: Arrow width
    """
    # Draw the line
    draw.line([start, end], fill=color, width=width)

    # Calculate the arrow head
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1:
        return

    # Normalize direction
    dx /= length
    dy /= length

    # Calculate arrow head points
    arrow_size = 10
    p1 = (end[0] - arrow_size * dx + arrow_size * dy / 2,
          end[1] - arrow_size * dy - arrow_size * dx / 2)
    p2 = (end[0] - arrow_size * dx - arrow_size * dy / 2,
          end[1] - arrow_size * dy + arrow_size * dx / 2)

    # Draw arrow head
    draw.polygon([end, p1, p2], fill=color)

def draw_node(draw, font, node, position):
    """
    Draw a node as a colored circle with text.

    Args:
        draw: PIL ImageDraw object
        font: PIL ImageFont object
        node: Node data
        position: (x, y) position
    """
    x, y = position

    # Determine node size based on file/dir count
    if node["type"] == "directory":
        size = max(30, min(80, 30 + node["file_count"] + node["dir_count"] * 2))
        # Choose a color based on depth
        color_idx = node["depth"] % len(DIRECTORY_COLORS)
        color = DIRECTORY_COLORS[color_idx]
        # Create label
        label = f"{node['name']}\n({node['file_count']} files, {node['dir_count']} dirs)"
    else:
        size = 20  # Fixed size for files
        # Choose a color based on file extension
        ext = os.path.splitext(node["name"])[1].lower()
        color_idx = hash(ext) % len(FILE_COLORS)
        color = FILE_COLORS[color_idx]
        label = node["name"]

    # Draw the circle
    draw.ellipse((x - size, y - size, x + size, y + size), fill=color, outline=(0, 0, 0))

    # Draw the text
    # Calculate text size
    text_width = font.getbbox(label)[2]
    text_lines = label.count('\n') + 1
    text_height = font.getbbox(label)[3] * text_lines

    # Position text
    text_x = x - text_width // 2
    text_y = y - text_height // 2

    # Draw text with outline for better visibility
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((text_x + dx, text_y + dy), label, fill=(0, 0, 0), font=font)
    draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze project structure
    project_root = Path(args.project_root)
    structure = analyze_directory(
        project_root,
        max_depth=args.max_depth + 1,  # Analyze one level deeper than visualization
        exclude_dirs=set(args.exclude_dirs)
    )

    # Save structure as JSON
    json_output_path = os.path.join(args.output_dir, "project_structure.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2)
    print(f"Project structure saved to {json_output_path}")

    # Create enhanced visualization with circles and arrows
    png_output_path = os.path.join(args.output_dir, "project_structure_colorful.png")
    create_visualization(
        structure,
        png_output_path,
        max_depth=args.max_depth,
        width=args.width,
        height=args.height
    )
    print(f"Enhanced visualization with circles and arrows saved to {png_output_path}")

    # Also create the original simple visualization for comparison
    simple_png_output_path = os.path.join(args.output_dir, "project_structure_simple.png")
    # We'll keep the original draw_tree function for the simple visualization
    img = Image.new('RGB', (args.width, args.height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
        title_font = font

    draw.text((20, 20), f"Project Structure: {structure['name']}", fill=TEXT_COLOR, font=title_font)
    draw_tree(draw, font, structure, 20, 80, 400, 40, args.max_depth)
    img.save(simple_png_output_path)
    print(f"Simple visualization saved to {simple_png_output_path}")

if __name__ == "__main__":
    main()
