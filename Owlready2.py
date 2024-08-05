import csv
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF

# Load your existing ontology
ontology_file = 'onto-tdm.owl'
g = Graph()
g.parse(ontology_file)

# Define the namespaces used in your ontology
ontology_ns = Namespace('http://www.owl-ontologies.com/Ontology1309777211.owl#') #change the namespace based on your specific ontology
rdf_ns = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')

# Load the instances from the CSV file
csv_file = 'output.csv' #use your specific csv containing annotated data to be used for ontology population

with open(csv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row

    # Iterate over each row in the CSV file
    for row in reader:
        notion_1 = row[0]
        notion_2 = row[1]
        knowledge_item = row[2]

        # Check if the instances already exist in the ontology
        if (ontology_ns[notion_1.replace(" ", "_")], RDF.type, ontology_ns.Notion) in g:
            notion_1_individual = ontology_ns[notion_1.replace(" ", "_")]
        else:
            # If the instance doesn't exist, create it and add appropriate type and superclass
            notion_1_individual = ontology_ns[notion_1.replace(" ", "_")]
            g.add((notion_1_individual, RDF.type, ontology_ns.Notion))
            g.add((notion_1_individual, ontology_ns.composed_of1, ontology_ns[notion_2.replace(" ", "_")]))

        if (ontology_ns[notion_2.replace(" ", "_")], RDF.type, ontology_ns.Notion) in g:
            notion_2_individual = ontology_ns[notion_2.replace(" ", "_")]
        else:
            # If the instance doesn't exist, create it and add appropriate type and superclass
            notion_2_individual = ontology_ns[notion_2.replace(" ", "_")]
            g.add((notion_2_individual, RDF.type, ontology_ns.Notion))

            g.add((notion_2_individual, ontology_ns.composed_of2, ontology_ns[knowledge_item.replace(" ", "_")]))

        if (ontology_ns[knowledge_item.replace(" ", "_")], RDF.type, ontology_ns.KnowledgeItem) in g:
            knowledge_item_individual = ontology_ns[knowledge_item.replace(" ", "_")]
        else:
            # If the instance doesn't exist, create it and add appropriate type and superclass
            knowledge_item_individual = ontology_ns[knowledge_item.replace(" ", "_")]
            g.add((knowledge_item_individual, RDF.type, ontology_ns.Knowledge_item))


# Save the modified ontology with the added instances
modified_ontology_file = 'ontotdm_ontology.owl'
g.serialize(modified_ontology_file, format='xml')
