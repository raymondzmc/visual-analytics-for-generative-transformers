
export const clearColor = () => {
  d3.selectAll('.clickable')
    .style('outline', 'none')
    .style('background-color', '#eee');
}

export const renderColor = (inputColor, outputColor, selected, state) => {

  clearColor();

  if (selected === 'input') {
    d3.select(`#input-token-${state.selectedInput}`)
      .style("outline", 'thin solid red')
      .style("outline-width", '2px');
  } else {
    d3.select(`#output-token-${state.selectedOutput}`)
      .style("outline", 'thin solid red')
      .style("outline-width", '2px');
  } 
  let bg_color;

  inputColor.forEach((value, i) => {
      value = +value;
      if (value === 0) {
        bg_color = '#eee';
      } else {
        bg_color = d3.interpolateReds(value / 100);
      }
      d3.select(`#input-token-${i}`)
          .style('background-color', bg_color);
  })


  outputColor.forEach((value, i) => {
      value = +value;
      if (value === 0) {
        bg_color = '#eee';
      } else {
        bg_color = d3.interpolateReds(value / 100);
      }
      d3.select(`#output-token-${i}`)
          .style('background-color', bg_color);
  })
}

export const renderInstanceView = (inputTokens, outputTokens, attributions, inputContainer, outputContainer, state) => {
  let inputs = d3.select(inputContainer).selectAll('li.input-token').data(inputTokens);
  let outputs = d3.select(outputContainer).selectAll('li.output-token').data(outputTokens);
  let inputsEnter = inputs.enter().append('li');
  let outputsEnter = outputs.enter().append('li');
  d3.select('#input-container-title').text(`Input Tokens:`);
  d3.select('#output-container-title').text(`Output Tokens:`);
  let metadataList = d3.select('#instance-metadata-list');
  
  metadataList.selectAll('li').remove() 

  inputsEnter.merge(inputs)
    .transition(300)
    .text(d => d)
    .attr('id', (d, i) => `input-token-${i}`)
    .attr('class', (d, i) => `input-token clickable`)
    .style('background-color', '#eee');

  
      

  outputsEnter.merge(outputs)
      .transition(300)
      .text(d => d)
      .attr('id', (d, i) => `output-token-${i}`)
      .attr('class', (d, i) => `output-token clickable`)
      .style('background-color', '#eee'); 

  d3.selectAll('.clickable')
      .style('outline', 'none')
      .style('background-color', '#eee');

  inputsEnter.merge(inputs)
    .on('click', function() {
      console.log(state.interpretation, state.encoderAttentions)
      if (state.interpretation === 'attention' && state.encoderAttentions != null) {

        state.selectedOutput = null;
        state.selectedInput = d3.select(this).attr('id').split('-')[2];
        let colors = state.encoderAttentions[state.selectedInput];
        renderColor(colors, [], 'input', state);
      } else if (state.interpretation === 'attribution' && state.attributions != null) {
        state.selectedOutput = null;
        state.selectedInput = d3.select(this).attr('id').split('-')[2];
        let colors = state.attributions[state.selectedInput];
        renderColor(colors, [], 'input', state);
        // console.log(state.selectedInput)
      }
    });

  outputsEnter.merge(outputs)
    .on('click', function() {
      let selectedOutput = d3.select(this).attr('id').split('-')[2];
      if (state.interpretation === 'attention' && state.decoderHead != null) {
        state.selectedInput = null;
        state.selectedOutput = selectedOutput;
        let inputColors = state.crossAttentions[selectedOutput];
        let outputColors = state.decoderAttentions[selectedOutput];
        renderColor(inputColors, outputColors, 'output', state);
      } else if (state.interpretation === 'attribution' && state.attributions != null) {
        state.selectedInput = null;
        state.selectedOutput = selectedOutput;
        let inputColors = state.attributions[selectedOutput]['input'];
        let outputColors = state.attributions[selectedOutput]['output'];
        renderColor(inputColors, outputColors, 'output', state);
      }
    });
  

  inputs.exit().remove();
  outputs.exit().remove();
  return state;
}