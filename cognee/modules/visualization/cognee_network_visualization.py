import os
import json
import networkx

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.storage import LocalStorage

logger = get_logger()


async def cognee_network_visualization(graph_data, destination_file_path: str = None):
    nodes_data, edges_data = graph_data

    G = networkx.DiGraph()

    nodes_list = []
    color_map = {
        "Entity": "#f47710",
        "EntityType": "#6510f4",
        "DocumentChunk": "#801212",
        "TextSummary": "#1077f4",
        "TableRow": "#f47710",
        "TableType": "#6510f4",
        "ColumnValue": "#13613a",
        "default": "#D3D3D3",
    }

    for node_id, node_info in nodes_data:
        node_info = node_info.copy()
        node_info["id"] = str(node_id)
        node_info["color"] = color_map.get(node_info.get("type", "default"), "#D3D3D3")
        node_info["name"] = node_info.get("name", str(node_id))

        try:
            del node_info[
                "updated_at"
            ]  #:TODO: We should decide what properties to show on the nodes and edges, we dont necessarily need all.
        except KeyError:
            pass

        try:
            del node_info["created_at"]
        except KeyError:
            pass

        nodes_list.append(node_info)
        G.add_node(node_id, **node_info)

    edge_labels = {}
    links_list = []
    for source, target, relation, edge_info in edges_data:
        source = str(source)
        target = str(target)
        G.add_edge(source, target)
        edge_labels[(source, target)] = relation
        links_list.append({"source": source, "target": target, "relation": relation})

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v5.min.js"></script>
        <style>
            body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: linear-gradient(90deg, #101010, #1a1a2e); color: white; font-family: 'Inter', sans-serif; }

            svg { width: 100vw; height: 100vh; display: block; }
            .links line { stroke: rgba(255, 255, 255, 0.4); stroke-width: 2px; }
            .nodes circle { stroke: white; stroke-width: 0.5px; filter: drop-shadow(0 0 5px rgba(255,255,255,0.3)); }
            .node-label { font-size: 5px; font-weight: bold; fill: white; text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', sans-serif; pointer-events: none; }
            .edge-label { font-size: 3px; fill: rgba(255, 255, 255, 0.7); text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', sans-serif; pointer-events: none; }
        </style>
    </head>
    <body>
        <svg></svg>
        <script>
            var nodes = {nodes};
            var links = {links};

            var svg = d3.select("svg"),
                width = window.innerWidth,
                height = window.innerHeight;

            var container = svg.append("g");

            var simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).strength(0.1))
                .force("charge", d3.forceManyBody().strength(-275))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("x", d3.forceX().strength(0.1).x(width / 2))
                .force("y", d3.forceY().strength(0.1).y(height / 2));

            var link = container.append("g")
                .attr("class", "links")
                .selectAll("line")
                .data(links)
                .enter().append("line")
                .attr("stroke-width", 2);

            var edgeLabels = container.append("g")
                .attr("class", "edge-labels")
                .selectAll("text")
                .data(links)
                .enter().append("text")
                .attr("class", "edge-label")
                .text(d => d.relation);

            var nodeGroup = container.append("g")
                .attr("class", "nodes")
                .selectAll("g")
                .data(nodes)
                .enter().append("g");

            var node = nodeGroup.append("circle")
                .attr("r", 13)
                .attr("fill", d => d.color)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            nodeGroup.append("text")
                .attr("class", "node-label")
                .attr("dy", 4)
                .attr("text-anchor", "middle")
                .text(d => d.name);

            node.append("title").text(d => JSON.stringify(d));

            simulation.on("tick", function() {
                link.attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                edgeLabels
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2 - 5);

                node.attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                nodeGroup.select("text")
                    .attr("x", d => d.x)
                    .attr("y", d => d.y)
                    .attr("dy", 4)
                    .attr("text-anchor", "middle");
            });

            svg.call(d3.zoom().on("zoom", function() {
                container.attr("transform", d3.event.transform);
            }));

            function dragstarted(d) {
                if (!d3.event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(d) {
                d.fx = d3.event.x;
                d.fy = d3.event.y;
            }

            function dragended(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }

            window.addEventListener("resize", function() {
                width = window.innerWidth;
                height = window.innerHeight;
                svg.attr("width", width).attr("height", height);
                simulation.force("center", d3.forceCenter(width / 2, height / 2));
                simulation.alpha(1).restart();
            });
        </script>

        <svg style="position: fixed; bottom: 10px; right: 10px; width: 150px; height: auto; z-index: 9999;" viewBox="0 0 158 44" fill="none" xmlns="http://www.w3.org/2000/svg">
            <text
                x="50%"
                y="50%"
                text-anchor="middle"
                dominant-baseline="central"
                font-size="32"
                fill="#fff"
                font-family='"Noto Sans SC", "Microsoft YaHei", "PingFang SC", "Heiti SC", "WenQuanYi Micro Hei", sans-serif'
            >
                天外来物
            </text>
        </svg>
    </body>
    </html>
    """

    html_content = html_template.replace("{nodes}", json.dumps(nodes_list)).replace("{links}", json.dumps(links_list))

    if not destination_file_path:
        home_dir = os.path.expanduser("~")
        destination_file_path = os.path.join(home_dir, "graph_visualization.html")

    LocalStorage.ensure_directory_exists(os.path.dirname(destination_file_path))

    with open(destination_file_path, "w") as f:
        f.write(html_content)

    logger.info(f"Graph visualization saved as {destination_file_path}")

    return html_content
