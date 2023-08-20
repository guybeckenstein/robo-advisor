from service.util import helpers


def save_graph(plt_instance, file_name) -> None:
    helpers.create_graphs_folders()
    # Adjust font size of the table cells
    plt_instance.savefig(f'{file_name}.png', format='png', transparent=True)
    plt_instance.clf()  # Clear the figure after saving