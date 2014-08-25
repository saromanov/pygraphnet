import pyparsing
#http://pyparsing.wikispaces.com/
#http://pythonhosted.org/pyparsing/

def play_with_element():
	element = pyparsing.ParserElement()
	element.setName('test')
	element.parseFile('pyparsing_fun.py')
	element.preParse('string for parsing', 1)


w = pyparsing.Word(pyparsing.alphas)
res = w.parseString('Chating with Batya Herrmann') #Chating
rule = pyparsing.anyOpenTag
#print(rule.parseString('<abcdef> <nofull>'))
q= pyparsing.QuotedString('afdf', escChar='a', escQuote='f')
print(q.parseString('afdf'))
