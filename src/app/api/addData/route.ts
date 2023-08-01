import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { NextRequest, NextResponse } from 'next/server';
import { PineconeClient } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';

const apiKey = process.env.OPENAI_API_KEY;

export async function POST(request: NextRequest) {
	const data = await request.formData();
	const file: File | null = data.get('file') as unknown as File;

	if (!file) {
		return NextResponse.json({ success: false, error: 'No file found' });
	}

	if (file.type !== 'application/pdf') {
		return NextResponse.json({ success: false, error: 'Invalid file type' });
	}

	const pdfLoader = new PDFLoader(file);
	const splitDocuments = await pdfLoader.loadAndSplit();

	const pineconeClient = new PineconeClient();
	await pineconeClient.init({
		apiKey: process.env.PINECONE_API_KEY ?? '',
		environment: 'gcp-starter',
	});
	const pineconeIndex = pineconeClient.Index(
		process.env.PINECONE_INDEX_NAME as string
	);

	await PineconeStore.fromDocuments(
		splitDocuments,
		new OpenAIEmbeddings(
			{
				openAIApiKey: apiKey!,
				modelName: 'vicuna-13b-v1.1',
			},
			{ basePath: 'https://shale.live/v1' }
		),
		{
			pineconeIndex,
		}
	);

	return NextResponse.json({ success: true });
}
