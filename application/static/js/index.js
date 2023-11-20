import { renderProjection, selectExample} from './projection-view.js';
import { renderImportanceFromState, renderImportance } from './component-view.js'
import { renderColor, clearColor} from './instance-view.js'

let state = {
  'corpus': 'sst_demo',
  'comparisonMode': false,
  'projectionType': 'hidden',
  'selectedExample': null,
  
    
  'type': 'train',
  'model': 'bert',
  'hiddenLayer': 12,

  'discrete': [],
  'continuous': [],

  'discreteFilters': {},
  'continuousFilters': {},
  
  'projectionColor': $("#color-select option:selected").val(),
  'canvasID': 'canvas',
  'filtersID': 'filter',

  'checkpointName': null,

  // For visualizing attention images
  'encoderAttentionView': $('#encoderAttentionViewSelect').find(':checked').val(),
  'encoderAttentionScale': $('#encoderAttentionScaleSelect').find(':checked').val(),
  'decoderAttentionView': $('#decoderAttentionViewSelect').find(':checked').val(),
  'decoderAttentionScale': $('#decoderAttentionScaleSelect').find(':checked').val(),

  'decoder_importance': null,
  'encoder_importance': null,
  'attributions': null,


  'encoder_attention': null,
  'decoder_attention': null,
  'cross_attention': null,

  
  'interpretation': $('#interpretation-select').find(':checked').val(),

  'pruned_heads': {},

  'comparisonState': null,

  // Seq2seq parameters
  'projectionMode': 'encoder',
  'decoderProjection': null,
  'responseData': null,

  'encoderHead': null,
  'decoderHead': null,
  'encoderAttentions': null,
  'crossAttentions': null,
  'decoderAttentions': null,

  'selectedInput': null,
  'selectedOutput': null,

  // customizable parameters
  'reload_umap_proj': false,
  'reload_tsne_proj': false,
  'n_neighbors': null,
  'min_dist': null,
  'n_components': null,
  'metric': null,
}

$(document).ready(function(){
  $(".tabs").click(function(){
      $(".tabs").removeClass("active");
      $(".tabs h6").removeClass("font-weight-bold");
      $(".tabs h6").addClass("text-muted");
      $(this).children("h6").removeClass("text-muted");
      $(this).children("h6").addClass("font-weight-bold");
      $(this).addClass("active");
      let current_fs = $(".active");
      let next_fs = $(this).attr('id');
      next_fs = "#" + next_fs + "-tab";
      $("fieldset").removeClass("show");
      $(next_fs).addClass("show");
      current_fs.animate({}, {
        step: function() {
          current_fs.css({
          'display': 'none',
          'position': 'relative'
          });
          next_fs.css({
            'display': 'block'
          });
        }
      });
  });
});
$('#projectionTabs > li > a').click(function() {
  if (state.projectionType !== $(this).attr('value')) {
    state.projectionType = $(this).attr('value');

    if (state.projectionType === 'hidden') {
      $('#hiddenLayer').removeClass('disabled');
    } else {

      $('#hiddenLayer').addClass('disabled');
    }

    state = loadData(state);
  }
});

// Attention View
$("#encoderAttentionView").change(function(){
  let value = $(this).find(':checked').val();
  state.attentionView = value;
  if (value === 'importance') {
    console.log("hoooo")
    d3.selectAll('.attn-pattern').attr('visibility', 'hidden');
  } else {
    d3.selectAll('.attn-pattern').attr('visibility', 'visible');
  }
});

$('#attention-comparison-select').change(function(){
  let value = $(this).find(':checked').val();
  if (value == 'right') {
    state.comparisonState = renderImportanceFromState(attentionSVG, state.comparisonState);
  } else {
    state = renderImportanceFromState(attentionSVG, state.comparisonState);
  }
})


$("#comparison-mode").change(function(){
  let value = $(this).find(':checked').val();
  let compare = (value === 'on')? true : false;
  state.comparisonMode = compare;
  
  if (compare) {
    $('#attention-right').prop("disabled", false);
    $('#interpretation-right').prop("disabled", false);
    $('#projection-options').hide();
    $('#projection-options-compare').show();
    state.comparisonState = Object.assign({}, state);
    state.comparisonState.canvasID = state.canvasID.replace('Left', 'Right');
    state.comparisonState.filtersID = state.filtersID.replace('left', 'right');

    $(`#${state.comparisonState.canvasID}`).show();
    $(`#${state.comparisonState.filtersID}`).show();

    state.comparisonState = loadData(state.comparisonState);

  } else {
    $('#attention-right').prop("disabled", true);
    $('#interpretation-right').prop("disabled", true);
    $('#projection-options').show();
    $('#projection-options-compare').hide();
    $(`#${state.comparisonState.canvasID}`).hide();
    $(`#${state.comparisonState.filtersID}`).hide();
    state.comparisonState = null;
  }

  state = loadData(state);
  
});



$('#reset-heads').on('click', function(event) {
  state.pruned_heads = {};

  if (state.selectedIdx.size !== 0) {
    let it = state.selectedIdx.values()
    let idx = it.next().value
    state = selectExample(idx, state);
    state = renderImportanceFromState(attentionSVG, state);
  } else {
    state = renderImportanceFromState(attentionSVG, state);
  }

});

$('#compute-instance').on('click', function(event) {
  console.log(state['example_id']);
  if (state['example_id'] !== undefined) {
    state = selectExample(state['example_id'], state);
  } else {
    state = renderImportanceFromState(attentionSVG, state);
  }

});


// Instance Investigation View
$("#interpretation-select").change(function(){
  let value = $(this).find(':checked').val();
  state.interpretation = value;

  // TODO: Redundant with function in selectToken() in "component-view.js"
  if (value === 'attention') {
    if (state.selectedOutput !== null && state.decoderAttentions !== null) {
      renderColor(state.crossAttentions[state.selectedOutput], state.decoderAttentions[state.selectedOutput], 'output', state);
    } else {
      clearColor();
    }
  } else if (value === 'attribution') {
    state.selectedInput = null;
    if (state.selectedOutput !== null && state.attributions !== null) {
      renderColor(state.attributions[state.selectedOutput]['input'], state.attributions[state.selectedOutput]['output'], 'output', state);
    } else {
      clearColor();
    }
  }
});


// Projection View

const projectionWidth = 1300;
const projectionHeight = 500;
const projectionSVG = d3.select("#projectionView")
  .append('svg')
  .attr("class", "scatterplot")
  .attr("width", projectionWidth)
  .attr("height", projectionHeight);

state.projectionSVG = projectionSVG;
state.projectionWidth = projectionWidth;
state.projectionHeight = projectionHeight;


const projectionCanvas = d3.select("#projectionView")
  .append('canvas')
    .attr('id', state.canvasID);


const decoderAttentionSVG = d3.select("#decoderAttentionView")
  .append('svg')
    .attr("value", "decoder")
    .attr("id", "decoder-attention-svg")
    .attr("height", 630);

const encoderAttentionSVG = d3.select("#encoderAttentionView")
  .append('svg')
    .attr("value", "encoder")
    .attr("id", "encoder-attention-svg")
    // .attr("width", 700)
    .attr("height", 630);



const tokenList = d3.select("#instanceView")
  .attr("height", 1400);
 
$("#reloadUMAP").click(function(){
  state.n_neighbors = parseInt(document.getElementById("n_neighbors").value)
  state.min_dist = parseFloat(document.getElementById("min_dist").value)
  state.n_components = parseInt(document.getElementById("umap_n_components").value)
  state.metric = document.getElementById("umap_select-metric").value
  state.reload_umap_proj = true
  state = loadData(state) 
  $('#projModal').modal('hide');
});

$("#reloadTSNE").click(function(){
  state.perplexity = parseInt(document.getElementById("perplexity").value)
  state.n_components = parseInt(document.getElementById("tsne_n_components").value)
  state.metric = document.getElementById("tsne_select-metric").value
  state.reload_tsne_proj = true
  state = loadData(state) 
  $('#projModal').modal('hide');
});

const loadData = (state) => {
  $('#loader').show(); 
  const server_query = d3.json('../api/data', {
      method: "POST",
      body: JSON.stringify(state),
      headers: {
          "Content-type": "application/json; charset=UTF-8"
      }
  })

  server_query.then(response => {

    let encoderImportance = response['encoder_head_importance'];
    let decoderImportance = response['decoder_head_importance'];
    let attn_patten = [];
 
    if (response['x'] === undefined || response['y'] === undefined) {
      $('#loader').hide();
      return;
    }

    if (encoderImportance !== undefined) {
      state = renderImportance(encoderImportance, attn_patten, encoderAttentionSVG, 850, 630, state);
    }

    if (decoderImportance !== undefined) {
      state = renderImportance(decoderImportance, attn_patten, decoderAttentionSVG, 850, 630, state);
    }

    state.encoder_importance = encoderImportance;
    state.decoder_importance = decoderImportance;
    state.aggregate_pattern = attn_patten;



    
    state.responseData = response;
    state = renderProjection(response, projectionSVG, projectionWidth, projectionHeight, 'encoder', state);
    $('#loader').hide();

  });
  return state
}




if (!state.reload_umap_proj && !state.reload_tsne_proj) {
  state = loadData(state);
}



