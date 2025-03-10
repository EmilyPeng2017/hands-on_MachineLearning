from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# visualize the trained decision tree by first using the export_graphviz() function to output a graph definition file called iris_tree.dot
from sklearn.tree import export_graphviz
# use graphviz.Source.from_file() to load and display the file in a Jupyter notebook
from graphviz import Source

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

# Define file paths
dot_filename = "iris_tree.dot"
png_filename = "iris_tree"

export_graphviz(
    tree_clf,
    out_file=dot_filename,
    feature_names=["petal length (cm)", "petal width (cm)"],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

graph = Source.from_file(dot_filename)
# Convert to PNG and save
graph.render(png_filename, format="png")