import { NextRequest } from "next/server";
import { Message as VercelChatMessage, StreamingTextResponse, LangChainStream, Message } from "ai";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { BytesOutputParser,StringOutputParser } from "langchain/schema/output_parser";
import { PromptTemplate } from "langchain/prompts";
// import { RunnableSequence, RunnablePassthrough } from "langchain/schema/runnable";

import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";

import { Document } from "langchain/document";
import {ConversationalRetrievalQAChain} from "langchain/chains";
import {  ChatMessageHistory, ConversationSummaryMemory } from "langchain/memory";
import { CallbackManager } from "langchain/callbacks";

export const runtime = "edge";

// type ConversationalRetrievalQAChainInput = {
//   question: string;
//   chat_history: VercelChatMessage[];
// };


const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

// const combineDocumentsFn = (docs: Document[], separator = "\n\n") => {
//   const serializedDocs = docs.map((doc) => doc.pageContent);
//   return serializedDocs.join(separator);
// };


const getVercelChatHistory = (docs: VercelChatMessage[]) => {
  const formattedDialogueTurns = docs.map((message) => {
    if (message.role === "user") {
      return `Human: ${message.content}`;
    } else if (message.role === "assistant") {
      return `Assistant: ${message.content}`;
    } else {
      return `${message.role}: ${message.content}`;
    }
  });

  return formattedDialogueTurns.join("\n");
};

// // This is used to add previous messages to the chat history.
const formatVercelMessages = (chatHistory: VercelChatMessage[]) => {

  const formattedDialogueTurns = chatHistory.map((message) => {
    if (message.role === "user") {
      return `Human: ${message.content}`;
    } else if (message.role === "assistant") {
      return `Assistant: ${message.content}`;
    } else {
      return `${message.role}: ${message.content}`;
    }
  });

  return formattedDialogueTurns.join("\n");
};


const CONDENSE_QUESTION_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History: 
{chat_history}
Follow up input: {question}
Standalone Question:`;

  // const condenseQuestionPrompt = new PromptTemplate({
  //   template: CONDENSE_QUESTION_TEMPLATE,
  //   inputVariables: ["chat_history", "question"],
  // });


// // #3
const ANSWER_TEMPLATE = `You are a receptionist bot at "Four Points Hotel".Answer the question according to the context in the most polite way possible.

Answer the question based only on the following context:
{context}

Question: {question}
`;


const answerPrompt = new PromptTemplate({
  template : ANSWER_TEMPLATE,
  inputVariables: ["context", "question"],

});


// export async function POST(req: NextRequest) {


//   // console.log("we are inside Route");

//   const body = await req.json();
//   const messages = body.messages ?? [];
//   const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
//   const currentMessageContent = messages[messages.length - 1].content;




//   // *_____CREATING CLIENT (SUPABASE) AND CALLING IN THE MODEL___

  // const model = new ChatOpenAI({
  //   temperature: 0.5,
  //   modelName: "gpt-3.5-turbo",
  //   stop: undefined,
  // });

  // const client = createClient(
  //   process.env.SUPABASE_URL!,
  //   process.env.SUPABASE_PRIVATE_KEY!
  // );
  
  // const vectorstore = new SupabaseVectorStore(new OpenAIEmbeddings(), {
  //   client,
  //   tableName: "newHotelDetails",
  //   // queryName: "match_documents",
  // });

//   const retriever = vectorstore.asRetriever();

//  // *_____HANDLE MEMORY AND PLAT WITH CHAINS___


//   const memory = new ConversationSummaryMemory({
//     llm: model, 
//     memoryKey: "chat_history", 
//     // inputKey: "question", 
//     // outputKey: "answer", 
//     returnMessages: true
//   });



//   const standaloneQuestionChain = RunnableSequence.from([
//     {
//       question: (input: ConversationalRetrievalQAChainInput) =>
//         input.question,
//       chat_history: (input: ConversationalRetrievalQAChainInput) =>
//             // memory
//             input.chat_history,
//         // formatVercelMessages(input.chat_history)
//     },
//     condenseQuestionPrompt,
//     model,
//     new StringOutputParser(),
//   ]);


//   // console.log(standaloneQuestionChain)
//   const answerChain = RunnableSequence.from([
//     {
//       context: retriever.pipe(combineDocumentsFn),
//       question: new RunnablePassthrough(),
//     },
//     answerPrompt,
//     model,
//     new BytesOutputParser(),
//   ]);
//   // const answerChain = RunnableSequence.from([
//   //   {
//   //     context: retriever,
//   //     question: currentMessageContent
      
//   //   },
//   //   answerPrompt,
//   //   model,
//   //   new BytesOutputParser(),
//   // ]);

//   // const conversationalRetrievalQAChain =
//   // answerChain;
//   const conversationalRetrievalQAChain =
//   standaloneQuestionChain.pipe(answerChain);

  // const stream = await conversationalRetrievalQAChain.stream({
  //   question: currentMessageContent,
  //   chat_history: formattedPreviousMessages.join("\n"),
  // });


  // // console.log("::::::: THE END :::::::\n\n" )
  // // const chain = prompt.pipe(model).pipe(outputParser);


  
  //   return new StreamingTextResponse(stream);
  // }

  
export async function POST(req: NextRequest) {

  const body = await req.json();
  const messages = body.messages ?? [];
  const { stream, handlers } = LangChainStream();

  const model = new ChatOpenAI({
    temperature: 0.5,
    modelName: "gpt-3.5-turbo",
    streaming: true,
  });
  
  const nonStreamingModel = new ChatOpenAI({
    temperature: 0,
  });

  const client = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_PRIVATE_KEY!
  );
  
  const vectorstore = new SupabaseVectorStore(new OpenAIEmbeddings(), {
    client,
    tableName: "newHotelDetails",
    queryName: "match_documents",
  });

  const memory = new ConversationSummaryMemory({
      llm: model, 
      memoryKey: "context", 
      inputKey: "question", 
      outputKey: "answer", 
      returnMessages: true,
    });
  
  // const previousMessages = messages.slice(0, -1);
  
  const chatHist = (body.messages ?? []).filter(
    (message: VercelChatMessage) =>
      message.role === "user" || message.role === "assistant",
  );
  const qachain =  ConversationalRetrievalQAChain.fromLLM(
    model, 
    vectorstore.asRetriever(),{
    memory: memory,
    qaChainOptions: {
      type: "stuff", 
      prompt: PromptTemplate.fromTemplate(ANSWER_TEMPLATE)
    },
    questionGeneratorChainOptions:{
      template: CONDENSE_QUESTION_TEMPLATE, 
      llm:nonStreamingModel
    },
    // qaTemplate: ANSWER_TEMPLATE,
  });

  const callbacks = CallbackManager.fromHandlers(handlers);
  const latest = messages[messages.length - 1].content;
  // (messages as Message[]).map((m) => `${m.role}: ${m.content}`).join('\n') 
  qachain.call({ question: latest, chat_history:getVercelChatHistory(chatHist)}, callbacks).catch((e) => {
    console.error("THIS IS THE THING ____", e);
  }); 
  // stream.add("HI YOOO")
  // console.log(getVercelChatHistory(chatHist));
  console.log(memory.buffer);
  // console.log("_____WE ARE AT THE END OF ROUTE _____");
  return new StreamingTextResponse(stream);
}