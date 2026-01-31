import pandas as pd
from tqdm import tqdm

input_file = './all_events.csv'
output_file = './all_events_formatted.csv'

if __name__ == '__main__':
    """ Reads previously generated `all_events.csv` file and changes it to the appropriate format """
    
    print('\nREFORMATTING EVENTS')
    with open(input_file, 'r') as file:
        lines = [line.rstrip() for line in file]
        
        headers = []
        data = []
        for line in tqdm(   lines,
                            desc='Events reformatted',
                            unit='rows' ):
            line : str

            if len(headers) == 0:
                headers = line.split(',')
                continue

            line = line.replace('|',"\"")
            line = line.replace('\'',"")
            line = line.replace(', ',",")
            line = line.split(',')

            line_data = []
            bracket_found = False
            str_element = ""
            for line_element in line:
                if "[" in line_element:
                    str_element += line_element
                    bracket_found = True
                elif "]" in line_element and bracket_found:
                    str_element += f',{line_element}'
                    str_element = str_element.replace("\"","")
                    line_data.append(str(str_element))
                    bracket_found = False
                elif bracket_found:
                    str_element += f',{line_element}'

                else:
                    line_data.append(float(line_element))

            if len(line_data) > len(headers):
                line_data.pop(len(line_data) - 2)

            data.append(line_data)
        
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(output_file, index=False)

    print("DONE")