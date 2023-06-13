import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import request from "request";

function parseDocumentRes(data, embeddings) {
    const parsedData = [];
    for (let i = 0; i < data.length; i++) {
        const text = data[i];
        const vector = embeddings[i];
        parsedData.push({ text, vector });
    }
    return parsedData;
}
function wrapAndSort(queryVector, dataVector) {
    const wrappedData = [];
    dataVector.forEach(data => {
      const similarity = cosineSimilarity(data.vector, queryVector);
      wrappedData.push({similarity, "text": data.text});
    });
    wrappedData.sort((a, b) => b.similarity - a.similarity); // Sort in descending order
    return wrappedData;
}

function cosineSimilarity(vectorA, vectorB) {
    if (vectorA.length !== vectorB.length || vectorA.length === 0) {
    return 0; // Handle edge cases
    }

    const dotProduct = vectorA.reduce((sum, value, index) => sum + value * vectorB[index], 0);
    const magnitudeA = Math.sqrt(vectorA.reduce((sum, value) => sum + value * value, 0));
    const magnitudeB = Math.sqrt(vectorB.reduce((sum, value) => sum + value * value, 0));

    if (magnitudeA === 0 || magnitudeB === 0) {
    return 0; // Handle edge cases
    }

    return dotProduct / (magnitudeA * magnitudeB);
}

async function main(){

    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY
    });
    const dataFaq = ["Hello","Wold"]
    const documentRes = await embeddings.embedDocuments(dataFaq);
    const dataFaqVector = parseDocumentRes(dataFaq,documentRes)
    const queryResp = await embeddings.embedQuery("how to do refund");
    console.log(wrapAndSort(queryResp,dataFaqVector))
}   

main()
