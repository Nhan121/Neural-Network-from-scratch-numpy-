import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

f = "0"
p = "1"
input1 = [1., 0., 0., 1.]
input2 = [1., 1, 0., 0.]
cm = sns.light_palette("lightgreen", as_cmap=True)

def AND_perceptron_sims():
    """
        This 
    """
    and_pctn = pd.DataFrame({'input 1': np.int8(input1),
                             'input 2': np.int8(input2),
                             'output: AND(inp1, inp2)': [int(input1[k]) & int(input2[k]) for k in range(4)]
                            })
    
    display(and_pctn.style.background_gradient(cmap=cm))
    fig = go.Figure(px.scatter(x = input1, y = input2, color = [p, f, f, f],
                              labels = {
                                         "x": "input 1",
                                         "y": "input 2",
                                         "color": "AND (input1, input2)"
                     }
                    ))
    fig.add_trace(go.Scatter(x = [1., 0., 1], y = [1., 1, 0.], mode='lines', 
                             line = dict(width = 0.5, color = 'rgb(184, 247, 212)'),
                             name = 'area1', text = 'int(linear_combination) > 0', fill='toself'))
    fig.add_trace(go.Scatter(x = [0., 1, 0.], y = [1., 0, 0], mode='lines', 
                             line = dict(width = 0.5),
                             name = 'area2', text = 'int(linear_combination) <=0', fill='toself'))
    fig.update_layout(width = 530, height = 400, 
                      xaxis_range = (-0.1, 1.1), yaxis_range = (-0.1, 1.1), title = "AND PERCEPTRON")
    fig.show()
    
def test_logical_perceptron(weight1, weight2, bias, logic_oper_test = 'AND'):
    
    # DON'T CHANGE ANYTHING BELOW
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    if logic_oper_test == 'AND':
        correct_outputs = [np.logical_and(x[0], x[1]) for x in test_inputs] 
    elif logic_oper_test == 'OR':
        correct_outputs = [np.logical_or(x[0], x[1]) for x in test_inputs]
    elif logic_oper_test == 'XOR':
        correct_outputs = [np.logical_xor(x[0], x[1]) for x in test_inputs]
    elif logic_oper_test == 'NAND':
        correct_outputs = [~np.logical_and(x[0], x[1]) for x in test_inputs]
    elif logic_oper_test == 'NOR':
        correct_outputs = [~np.logical_or(x[0], x[1]) for x in test_inputs]
    elif logic_oper_test == 'XNOR':
        correct_outputs = [~np.logical_xor(x[0], x[1]) for x in test_inputs]    
    else:
        raise TypeError("The valid of the parameters logic_oper_test must be {AND, OR, NAND, XOR, XNOR, NOR}")
    
    print("True output: {}".format(correct_outputs))
    
    outputs = []

    # Generate and check output
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output = int(linear_combination >= 0)
        is_correct_string = 'Yes' if output == int(correct_output) else 'No'
        outputs.append([test_input[0], test_input[1], linear_combination, output, 
                        int(correct_output), is_correct_string])

    # Print output
    res = [output[5] for output in outputs if output[5] == 'No']
    num_wrong = len(res)

    output_frame = pd.DataFrame(outputs, columns=['Input 1', 'Input 2', 'Linear Combination', 
                                                  'Activation Output', 'correct_output', 'Is Correct'])
    if num_wrong < 1:
        print('Nice!  You got it all correct.\n')
    else:
        print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
    output_frame.style.hide_index()
    
    return (output_frame.style.background_gradient(cmap=cm).set_precision(2).highlight_null('lightred').highlight_min(axis=0, color='lightblue').highlight_max(axis=0, color='red'))
    
def OR_perceptron_sims():
    """
    
    """
    or_pctn = pd.DataFrame({'input 1': np.int8(input1),
                             'input 2': np.int8(input2),
                             'OR (inp1, inp2)': [int(input1[k]) | int(input2[k]) for k in range(4)]
                            })
    cm = sns.light_palette("lightgreen", as_cmap=True)
    fig = go.Figure(px.scatter(x = input1, y = input2, color = [p, p, f, p],
                              labels = {
                                         "x": "input 1",
                                         "y": "input 2",
                                         "color": "OR (input1, input2)"
                     }
                    ))
    fig.add_trace(go.Scatter(x = [1., 0., 1], y = [1., 1, 0.], mode='lines', 
                             line = dict(width = 0.5, color = 'rgb(184, 247, 212)'),
                             name = 'area1', text = 'int(linear_combination) >= 0', fill='toself'))
    fig.add_trace(go.Scatter(x = [0., 1, 0.], y = [1., 0, 0], mode='lines', 
                             line = dict(width = 0.5),
                             name = 'area2', text = 'int(linear_combination) < 0', fill='toself'))
    fig.update_layout(width = 500, height = 400, 
                      xaxis_range = (-0.1, 1.1), yaxis_range = (-0.1, 1.1), title = "OR PERCEPTRON")
    fig.show()
    display(or_pctn.style.background_gradient(cmap=cm))
    
def XOR_perceptron_sims():
    """
    
    """
    or_pctn = pd.DataFrame({'input 1': np.int8(input1),
                             'input 2': np.int8(input2),
                             'XOR (inp1, inp2)': [int(np.logical_xor(input1[k], input2[k])) for k in range(4)]
                            })
    cm = sns.light_palette("lightgreen", as_cmap=True)
    fig = go.Figure(px.scatter(x = input1, y = input2, color = [f, p, f, p],
                              labels = {
                                         "x": "input 1",
                                         "y": "input 2",
                                         "color": "XOR (input1, input2)"
                     }
                    ))
    fig.update_layout(width = 500, height = 400, 
                      xaxis_range = (-0.1, 1.1), yaxis_range = (-0.1, 1.1), title = "XOR PERCEPTRON")
    fig.show()
    display(or_pctn.style.background_gradient(cmap=cm))
    
cm = sns.light_palette("lightgreen", as_cmap=True)
def XOR_perceptron_sims():
    """
    
    """
    or_pctn = pd.DataFrame({'input 1': np.int8(input1),
                             'input 2': np.int8(input2),
                             'XOR (inp1, inp2)': [int(np.logical_xor(input1[k], input2[k])) for k in range(4)]
                            })
    cm = sns.light_palette("lightgreen", as_cmap=True)
    fig = go.Figure(px.scatter(x = input1, y = input2, color = [f, p, f, p],
                              labels = {
                                         "x": "input 1",
                                         "y": "input 2",
                                         "color": "XOR (input1, input2)"
                     }
                    ))
    fig.add_trace(go.Scatter(x = [0.0, -0.5, -.5, 1.1, 1.1, 1., -0.1], 
                             y = [-.1, -.1, 1.5, 1.5, 1., 1., 1.1],
                             mode='lines', 
                             line = dict(width = 0.5, color = 'rgb(184, 247, 212)'),
                             name = 'area1', text = 'nonlinear region', fill='toself'))
    
    fig.update_layout(width = 500, height = 400, 
                      xaxis_range = (-0.5, 1.1), yaxis_range = (-0.1, 1.5), title = "XOR PERCEPTRON")
    fig.show()
    display(or_pctn.style.background_gradient(cmap=cm))
