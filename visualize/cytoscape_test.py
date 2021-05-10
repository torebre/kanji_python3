from py2cytoscape.data.cyrest_client import CyRestClient
import networkx as nx

from cluster_graph.functions_for_creating_cluster_graph import create_graph_showing_number_of_paths
from cluster_graph.setup_cluster_graph import setup_graph_type_2
import py2cytoscape.cytoscapejs as renderer

if __name__ == '__main__':
    cy = CyRestClient()
    network = cy.network.create(name='My Network', collection='My network collection')

    print(network.get_id())

    cluster_graph: nx.MultiDiGraph = setup_graph_type_2()
    edge_graph: nx.DiGraph = create_graph_showing_number_of_paths('1', cluster_graph)
    cy.network.create_from_networkx(edge_graph)

    cy.layout.apply(name='force-directed', network=cluster_graph)

    network_style = cy.style.create('GAL Style')

    basic_settings = {
        'NODE_FILL_COLOR': '#6AACB8',
        'NODE_SIZE': 55,
        'NODE_BORDER_WIDTH': 0,
        'NODE_LABEL_COLOR': '#555555',

        'EDGE_WIDTH': 2,
        'EDGE_TRANSPARENCY': 100,
        'EDGE_STROKE_UNSELECTED_PAINT': '#333333',

        'NETWORK_BACKGROUND_PAINT': '#FFFFEA'
    }

    network_style.update_defaults(basic_settings)

    yeast_net_view = network.get_first_view()
    style_for_widget = cy.style.get(network_style.get_name(), data_format='cytoscapejs')
    renderer.render(yeast_net_view, style=style_for_widget['style'],
                    background='radial-gradient(#FFFFFF 15%, #DDDDDD 105%)')
