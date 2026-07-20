from pullenti.ner.processor import Processor
from pullenti.ner.core import SourceOfAnalysis
from pullenti.ner.address import AddressReferent

processor = Processor([
    AddressReferent.OBJ_TYPENAME
])

text = """
Россия г Московская область р-н Троицк деревня Ботаково Заречная улица 117А
"""

result = processor.process(SourceOfAnalysis(text))

for entity in result.entities:
    print(type(entity), entity)