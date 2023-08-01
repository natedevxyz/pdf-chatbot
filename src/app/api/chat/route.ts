import { NextRequest } from 'next/server';
import { PineconeClient } from '@pinecone-database/pinecone';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { OpenAI } from 'langchain/llms/openai';
import { VectorDBQAChain } from 'langchain/chains';
import { StreamingTextResponse, LangChainStream } from 'ai';
import { CallbackManager } from 'langchain/callbacks';

const apiKey = process.env.OPENAI_API_KEY;

export async function POST(request: NextRequest) {
	const body = await request.json();

	const { stream, handlers } = LangChainStream();

	const pineconeClient = new PineconeClient();
	await pineconeClient.init({
		apiKey: process.env.PINECONE_API_KEY ?? '',
		environment: 'gcp-starter',
	});
	const pineconeIndex = pineconeClient.Index(
		process.env.PINECONE_INDEX_NAME as string
	);

	const vectorStore = await PineconeStore.fromExistingIndex(
		new OpenAIEmbeddings(
			{
				openAIApiKey: apiKey!,
				modelName: 'vicuna-13b-v1.1',
			},
			{ basePath: 'https://shale.live/v1' }
		),
		{ pineconeIndex }
	);

	const model = new OpenAI(
		{
			openAIApiKey: apiKey!,
			modelName: 'vicuna-13b-v1.1',
			streaming: true,
			callbackManager: CallbackManager.fromHandlers(handlers),
		},
		{ basePath: 'https://shale.live/v1' }
	);

	const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
		k: 1,
		returnSourceDocuments: true,
	});

	chain.call({ query: body.prompt }).catch(console.error);

	return new StreamingTextResponse(stream);
}
