本代码库Llama-VITS的具体实现，如果对你有所帮助，请考虑引用我们的论文"Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness". BibTeX: [@feng2024llamavits].



## Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness

<textarea id="copyText" style="width:620px; height:130px;">
@misc{feng2024llamavits,
      title={Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness}, 
      author={Xincan Feng and Akifumi Yoshimoto},
      year={2024},
      eprint={2404.06714},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</textarea>

<button onclick="copyTextFunction()">Copy BibTeX Citation</button>

<script>
function copyTextFunction() {
    var copyText = document.getElementById("copyText");
    copyText.select();
    copyText.setSelectionRange(0, 99999); // 移动设备兼容
    navigator.clipboard.writeText(copyText.value).then(function() {
        console.log('copied');
    }, function(err) {
        console.error('error: ', err);
    });
}
</script>
