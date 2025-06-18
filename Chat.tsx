import { useRef, useState, useEffect } from "react";
import { Checkbox, Panel, DefaultButton, TextField, SpinButton } from "@fluentui/react";
import { Dialog, DialogType, DialogFooter } from "@fluentui/react";
import { SparkleFilled, ChevronRightRegular, ChevronLeftRegular, InfoRegular, ContactCardRibbonRegular, PulseRegular, SaveEdit24Regular } from "@fluentui/react-icons";
import readNDJSONStream from "ndjson-readablestream";
import { jsonrepair } from "jsonrepair";
import { Dropdown } from "@fluentui/react";
import { useOutletContext } from "react-router-dom";
import UTACLogo from '../../assets/UTACLogo_ChatBOT.png';
import CollapseIcon from '../../assets/Sidebar.png';
import MonitorIcon from '../../assets/Monitor.png';
import styles from "./Chat.module.css";

import {
    chatApi,
    configApi,
    RetrievalMode,
    ChatAppResponse,
    ChatAppResponseOrError,
    ChatAppRequest,
    ResponseMessage,
    VectorFieldOptions,
    GPT4VInput,
    saveApi,
    SaveRequest,
    siteOptions,
    Country
} from "../../api";
import { Answer, AnswerError, AnswerLoading } from "../../components/Answer";
import { QuestionInput } from "../../components/QuestionInput";
import { ExampleList } from "../../components/Example";
import { UserChatMessage } from "../../components/UserChatMessage";
import { AnalysisPanel, AnalysisPanelTabs } from "../../components/AnalysisPanel";
import { SettingsButton } from "../../components/SettingsButton";
import { ClearChatButton } from "../../components/ClearChatButton";
import { useLogin, getToken, isLoggedIn, requireAccessControl, getUserId, appServicesToken } from "../../authConfig";
import { VectorSettings } from "../../components/VectorSettings";
import { useMsal } from "@azure/msal-react";
import { TokenClaimsDisplay } from "../../components/TokenClaimsDisplay";
import { GPT4VSettings } from "../../components/GPT4VSettings";
import { ClockRegular } from "@fluentui/react-icons";


interface DislikePopupProps {
    onClose: () => void;
    onSubmit: (reason: string) => void;
    isOpen: boolean;
}

const DislikePopup: React.FC<DislikePopupProps> = ({ onClose, onSubmit, isOpen }) => {
    const [reason, setReason] = useState("");

    const handleSubmit = () => {
        onSubmit(reason);
        onClose(); // Close modal on submit
    };

    return (
        <Dialog
            hidden={!isOpen}
            onDismiss={onClose}
            dialogContentProps={{
                type: DialogType.normal,
                title: "Please provide a reason for your dislike (optional):",
                subText: "Your feedback is important to us."
            }}
            modalProps={{
                isBlocking: false,
                styles: { main: { maxWidth: 450 } }
            }}
        >
            <TextField value={reason} onChange={(ev, newValue) => setReason(newValue || "")} multiline rows={3} />
            <DialogFooter>
                <DefaultButton onClick={handleSubmit} text="Submit" />
                <DefaultButton onClick={onClose} text="Cancel" />
            </DialogFooter>
        </Dialog>
    );
};

const cleanupDuplicateHistory = () => {
    const raw = localStorage.getItem('chatHistory');
    if (!raw) return;

    try {
        const parsed = JSON.parse(raw) as any[];
        const uniqueHistory = parsed.filter((item, index, arr) => {
            const question = Array.isArray(item) ? item[0] : item?.question;
            return arr.findIndex(other => {
                const otherQuestion = Array.isArray(other) ? other[0] : other?.question;
                return otherQuestion === question;
            }) === index;
        });

        localStorage.setItem('chatHistory', JSON.stringify(uniqueHistory));
    } catch (error) {
        console.error('Error cleaning history:', error);
    }
};

const Chat = () => {
    //const { selectedCountry, selectedSite } = useOutletContext<{ selectedCountry: string; selectedSite: string }>();
    const [isConfigPanelOpen, setIsConfigPanelOpen] = useState(false);
    const [promptTemplate, setPromptTemplate] = useState<string>("");
    const [retrieveCount, setRetrieveCount] = useState<number>(30);
    const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>(RetrievalMode.Vectors);
    const [useSemanticRanker, setUseSemanticRanker] = useState<boolean>(true);
    const [shouldStream, setShouldStream] = useState<boolean>(true);
    const [useSemanticCaptions, setUseSemanticCaptions] = useState<boolean>(false);

    const [excludeCategory, setExcludeCategory] = useState<string>("");
    const [useSuggestFollowupQuestions, setUseSuggestFollowupQuestions] = useState<boolean>(true);
    const [vectorFieldList, setVectorFieldList] = useState<VectorFieldOptions[]>([VectorFieldOptions.Embedding]);
    const [useOidSecurityFilter, setUseOidSecurityFilter] = useState<boolean>(false);
    const [useGroupsSecurityFilter, setUseGroupsSecurityFilter] = useState<boolean>(false);
    const [gpt4vInput, setGPT4VInput] = useState<GPT4VInput>(GPT4VInput.TextAndImages);
    const [useGPT4V, setUseGPT4V] = useState<boolean>(false);

    const lastQuestionRef = useRef<string>("");
    const chatMessageStreamEnd = useRef<HTMLDivElement | null>(null);

    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [isStreaming, setIsStreaming] = useState<boolean>(false);
    const [error, setError] = useState<unknown>();

    const [activeCitation, setActiveCitation] = useState<string>();
    const [activeAnalysisPanelTab, setActiveAnalysisPanelTab] = useState<AnalysisPanelTabs | undefined>(undefined);

    const [selectedAnswer, setSelectedAnswer] = useState<number>(0);
    const [answers, setAnswers] = useState<[user: string, response: ChatAppResponse][]>([]);
    const [streamedAnswers, setStreamedAnswers] = useState<[user: string, response: ChatAppResponse][]>([]);
    const [showGPT4VOptions, setShowGPT4VOptions] = useState<boolean>(false);
    const [isDevMode, setIsDevMode] = useState(false);
    const [likeStatus, setLikeStatus] = useState<"like" | "dislike" | null>(null);
    const [dislikeReason, setDislikeReason] = useState<string>("");
    const [showDislikePopup, setShowDislikePopup] = useState<boolean>(false);
    const [isPanelCollapsed, setIsPanelCollapsed] = useState(false); // State to track panel collapse
    const [collapsed, setCollapsed] = useState(false);
    // const [selectedCountry, setSelectedCountry] = useState<string>("");
    // const [selectedSite, setSelectedSite] = useState<string>("");
    const toggleCollapse = () => {
        setCollapsed(!collapsed);
    };
    enum Country {
        Singapore = "Singapore",
        US = "US",
        Switzerland = "Switzerland",
        Japan = "Japan",
        China = "China",
        Indonesia = "Indonesia",
        Thailand = "Thailand"
    }

    interface SiteOption {
        key: string;
        text: string;
    }

    type SiteOptionsMap = {
        [country: string]: SiteOption[];
    };

    const siteOptions: SiteOptionsMap = {
        Singapore: [
            { key: "USG", text: "USG" },
            { key: "Corporate", text: "Corporate" }
        ],
        US: [{ key: "UGS US", text: "UGS US" }],
        Switzerland: [{ key: "UGS SWT", text: "UGS SWT" }],
        Japan: [{ key: "UJP", text: "UJP" }],
        China: [
            { key: "UDGA", text: "UDGA" },
            { key: "UYT", text: "UYT" },
            { key: "USH", text: "USH" }
        ],
        Indonesia: [{ key: "UID", text: "UID" }],
        Thailand: [{ key: "UTL", text: "UTL" }]
    };
    const [isFirstChat, setIsFirstChat] = useState(true);  // Track if it's the first chat
    const [hasLoaded, setHasLoaded] = useState(false);
    const [selectedHistory, setSelectedHistory] = useState<{ question: string, response: ChatAppResponse } | null>(null);
    const [storedHistory, setHistory] = useState<{ question: string, response: ChatAppResponse }[]>([]);

    const { instance } = useMsal();
    const username = instance.getActiveAccount()?.username ?? appServicesToken?.user_claims?.preferred_username;

    const [selectedCountry, setSelectedCountry] = useState<Country>(Country.Singapore);
    const [selectedSite, setSelectedSite] = useState<string>(siteOptions[Country.Singapore][0].key);
    const [filteredSiteOptions, setFilteredSiteOptions] = useState<SiteOption[]>(siteOptions[Country.Singapore]);

    useEffect(() => {
        const accounts = instance.getAllAccounts();
        if (!instance.getActiveAccount() && accounts.length > 0) {
            instance.setActiveAccount(accounts[0]);
            console.log("Set MSAL active account to:", accounts[0].username);
        }
    }, [instance]);

    // 🔍 Log the values
    console.log("MSAL Active Account:", instance.getActiveAccount());
    console.log("Username from MSAL:", instance.getActiveAccount()?.username);
    console.log("Fallback Username from Token:", appServicesToken?.user_claims?.preferred_username);
    console.log("Final Username:", username);

    // Function to fetch and load the chat history from localStorage
    const loadHistoryFromLocalStorage = () => {
        const raw = localStorage.getItem('chatHistory');
        if (!raw) return;

        // Parse the stored value
        const parsed = JSON.parse(raw) as any[];

        // Normalize into objects { question, response }
        const normalized = parsed
            .map(entry => {
                // If entry already has .question, assume correct shape
                if (entry && typeof entry === 'object' && 'question' in entry) {
                    return entry as { question: string, response: ChatAppResponse };
                }
                // Otherwise, if entry is [question, response], convert it
                if (Array.isArray(entry) && entry.length === 2) {
                    return { question: entry[0], response: entry[1] } as { question: string, response: ChatAppResponse };
                }
                return null;
            })
            .filter(item => item !== null) as { question: string, response: ChatAppResponse }[];

        setHistory(normalized);
        // Note: Do NOT setAnswers(...) here, because this is only for history.
    };

    // THEN ADD this useEffect to clean duplicates on component mount:
    useEffect(() => {
        cleanupDuplicateHistory();
        loadHistoryFromLocalStorage();
    }, []);

    const handleHistoryClick = (userQuestion: string, response: ChatAppResponse) => {
        console.log('History clicked');
        // Update the selectedHistory with the clicked question and response
        setSelectedHistory({ question: userQuestion, response: response });
        setIsFirstChat(false);
        setHasLoaded(true);
    };

    useEffect(() => {
        const devMode = localStorage.getItem("dev") === "true"; // Assuming 'dev' key stores a string
        setIsDevMode(devMode);
    }, []);

    const getConfig = async () => {
        if (appServicesToken) {
            console.log("id token", appServicesToken.access_token);
            configApi(appServicesToken.access_token).then(config => {
                setShowGPT4VOptions(config.showGPT4VOptions);
            });
        }
    };


    useEffect(() => {
        const options = siteOptions[selectedCountry];
        setFilteredSiteOptions(options);
        setSelectedSite(options[0].key);
    }, [selectedCountry]);

    useEffect(() => {
        const getConfig = async () => {
            if (appServicesToken) {
                configApi(appServicesToken.access_token).then(config => {
                    if (config.ad_info && config.ad_info.country && config.ad_info.site) {
                        setSelectedCountry(Country[config.ad_info.country as keyof typeof Country]);
                        setSelectedSite(config.ad_info.site);
                    }
                });
            }
        };

        getConfig();
    }, []);

    const countryOptions = Object.values(Country).map(country => ({ key: country, text: country }));

    const handleAsyncRequest = async (question: string, answers: [string, ChatAppResponse][], setAnswers: Function, responseBody: ReadableStream<any>) => {
        let answer: string = "";
        let askResponse: ChatAppResponse = {} as ChatAppResponse;
        const updateState = (newContent: string) => {
            return new Promise(resolve => {
                setTimeout(() => {
                    answer += newContent;
                    const latestResponse: ChatAppResponse = {
                        ...askResponse,
                        choices: [{ ...askResponse.choices[0], message: { content: answer, role: askResponse.choices[0].message.role } }]
                    };
                    setStreamedAnswers([...answers, [question, latestResponse]]);
                    resolve(null);
                }, 33);
            });
        };
        try {
            setIsStreaming(true);

            for await (const event of readNDJSONStream(responseBody)) {
                if (event["choices"] && event["choices"][0]["context"] && event["choices"][0]["context"]["data_points"]) {
                    event["choices"][0]["message"] = event["choices"][0]["delta"];
                    askResponse = event;
                } else if (event["choices"] && event["choices"][0]["delta"]["content"]) {
                    setIsLoading(false);
                    await updateState(event["choices"][0]["delta"]["content"]);
                } else if (event["choices"] && event["choices"][0]["context"]) {
                    // Update context with new keys from latest event
                    askResponse.choices[0].context = { ...askResponse.choices[0].context, ...event["choices"][0]["context"] };
                } else if (event["error"]) {
                    throw Error(event["error"]);
                }
            }
        } finally {
            setIsStreaming(false);
        }
        const fullResponse: ChatAppResponse = {
            ...askResponse,
            choices: [{ ...askResponse.choices[0], message: { content: answer, role: askResponse.choices[0].message.role } }]
        };
        return fullResponse;
    };

    const client = useLogin ? useMsal().instance : undefined;

    const handleLike = async (index: number) => {
        setLikeStatus("like");
        console.log("like clicked");
        const userId = appServicesToken != null ? appServicesToken.user_id : undefined;
        const token = client ? await getToken(client) : undefined;
        // if (client) {
        console.log("like clicked. inside client", userId, token);
        // const userId = await getUserId(client);
        // const token = await getToken(client);
        const request: SaveRequest = {
            question: answers[index][0],
            answer: answers[index][1].choices[0].message.content,
            user_id: userId || undefined,
            country: selectedCountry,
            site: selectedSite,
            metadata: { likeStatus: "like" }
        };
        saveApi(request, token);
        // }
    };

    const makeApiRequest = async (question: string) => {
        setIsFirstChat(false);
        console.log(selectedHistory);
        setSelectedHistory(null);
        lastQuestionRef.current = question;

        error && setError(undefined);
        setIsLoading(true);
        setActiveCitation(undefined);
        setActiveAnalysisPanelTab(undefined);

        const token = client ? await getToken(client) : undefined;

        try {
            const messages: ResponseMessage[] = answers.flatMap(a => [
                // { content: a[0], role: "user" },
                //{ content: JSON.parse(jsonrepair(a[1].choices[0].context.thoughts.at(-1)?.description.at(-1))).content, role: "user" },
                { content: a[1].choices[0].message.content, role: "assistant" }
            ]);

            const request: ChatAppRequest = {
                messages: [...messages, { content: question, role: "user" }],
                stream: shouldStream,
                context: {
                    overrides: {
                        prompt_template: promptTemplate.length === 0 ? undefined : promptTemplate,
                        exclude_category: excludeCategory.length === 0 ? undefined : excludeCategory,
                        country: selectedCountry.length === 0 ? undefined : selectedCountry,
                        site: selectedSite.length === 0 ? undefined : selectedSite,
                        top: retrieveCount,
                        retrieval_mode: retrievalMode,
                        semantic_ranker: useSemanticRanker,
                        semantic_captions: useSemanticCaptions,
                        suggest_followup_questions: useSuggestFollowupQuestions,
                        use_oid_security_filter: useOidSecurityFilter,
                        use_groups_security_filter: useGroupsSecurityFilter,
                        vector_fields: vectorFieldList,
                        use_gpt4v: useGPT4V,
                        gpt4v_input: gpt4vInput
                    }
                },
                // ChatAppProtocol: Client must pass on any session state received from the server
                session_state: answers.length ? answers[answers.length - 1][1].choices[0].session_state : null
            };

            const response = await chatApi(request, token);
            //const data = await response.json(); // THIS gets the real JSON
            //console.log("Response chat JSON:", JSON.stringify(data, null, 2));
            if (!response.body) {
                throw Error("No response body");
            }
            if (shouldStream) {
                console.log("hi1")
                const parsedResponse: ChatAppResponse = await handleAsyncRequest(question, answers, setAnswers, response.body);

                // 1) Update answers state (unchanged)
                setAnswers([ ...answers, [question, parsedResponse] ]);

                // 2) Build an object entry for history
                const updatedEntry = { question: question, response: parsedResponse };
                // 3) Append to storedHistory array (which is the state for history)
                const newHistory = storedHistory.length > 0 
                    ? [...storedHistory, updatedEntry] 
                    : [updatedEntry];
                setHistory(newHistory);
                localStorage.setItem('chatHistory', JSON.stringify(newHistory));

                const userId = appServicesToken != null ? appServicesToken.user_id : undefined;
                console.log("hi1a")
                const request: SaveRequest = {
                    question: question,
                    answer: parsedResponse.choices[0].message.content,
                    user_id: userId,
                    country: selectedCountry,
                    site: selectedSite,
                    follow_up_questions: parsedResponse.choices[0].context.followup_questions
                };
                saveApi(request, token);
            } else {
                const parsedResponse: ChatAppResponseOrError = await response.json();
                if (response.status > 299 || !response.ok) {
                    throw Error(parsedResponse.error || "Unknown error");
                }

                // 1) Update answers (unchanged)
                setAnswers([...answers, [question, parsedResponse as ChatAppResponse]]);

                // 2) Build a new history‐entry object
                const updatedEntry = {
                    question: question,
                    response: parsedResponse as ChatAppResponse,
                };

                // 3) Append it to storedHistory (which was initialized as [])
                const newHistory = storedHistory.length > 0
                    ? [...storedHistory, updatedEntry]
                    : [updatedEntry];

                // 4) Update history state
                setHistory(newHistory);

                // 5) Persist the updatedHistory array of objects
                localStorage.setItem('chatHistory', JSON.stringify(newHistory));
            }
        } catch (e) {
            setError(e);

        } finally {
            setIsLoading(false);
        }
        console.log(isStreaming);
    };

    const clearChat = () => {
        console.log("Clearchat called");
        lastQuestionRef.current = "";
        error && setError(undefined);
        setActiveCitation(undefined);
        setActiveAnalysisPanelTab(undefined);
        setAnswers([]);
        setStreamedAnswers([]);
        setIsLoading(false);
        setIsStreaming(false);
        setIsFirstChat(true);
        setHasLoaded(false)
        setSelectedHistory(null);

    };

    const [keyboardHeight, setKeyboardHeight] = useState(0)
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

    // Detect screen size after first render
    useEffect(() => {
        const handleResize = () => {
            const mobile = window.innerWidth < 768;
            setIsMobile(mobile);
            setCollapsed(mobile); // collapsed = true for mobile
        };

        handleResize(); // run once on mount
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    useEffect(() => {
        if (!isMobile) return;
        
        // Method 1: Listen for window resize (works on most mobile browsers)
        const handleResize = () => {
            if (window.innerHeight < window.screen.height * 0.75) {
                // Keyboard is likely open
                setKeyboardHeight(window.screen.height - window.innerHeight);
            } else {
                // Keyboard is likely closed
                setKeyboardHeight(0);
            }
        };

        // Method 2: Listen for input focus/blur events
        const handleInputFocus = () => {
            setTimeout(() => {
                if (window.innerHeight < window.screen.height * 0.75) {
                    setKeyboardHeight(window.screen.height - window.innerHeight);
                }
            }, 300); // Small delay for keyboard animation
        };

        const handleInputBlur = () => {
            setTimeout(() => {
                setKeyboardHeight(0);
            }, 300);
        };

        // Add event listeners
        window.addEventListener('resize', handleResize);
        
        // Listen to all input elements in the chat
        const inputs = document.querySelectorAll('input, textarea');
        inputs.forEach(input => {
            input.addEventListener('focus', handleInputFocus);
            input.addEventListener('blur', handleInputBlur);
        });

        return () => {
            window.removeEventListener('resize', handleResize);
            inputs.forEach(input => {
                input.removeEventListener('focus', handleInputFocus);
                input.removeEventListener('blur', handleInputBlur);
            });
        };
    }, [isMobile]);

    useEffect(() => chatMessageStreamEnd.current?.scrollIntoView({ behavior: "smooth" }), [isLoading]);
    useEffect(() => chatMessageStreamEnd.current?.scrollIntoView({ behavior: "auto" }), [streamedAnswers]);
    useEffect(() => {
        getConfig();
    }, []);

    const onPromptTemplateChange = (_ev?: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string) => {
        setPromptTemplate(newValue || "");
    };

    const onRetrieveCountChange = (_ev?: React.SyntheticEvent<HTMLElement, Event>, newValue?: string) => {
        setRetrieveCount(parseInt(newValue || "3"));
    };

    const onUseSemanticRankerChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseSemanticRanker(!!checked);
    };

    const onUseSemanticCaptionsChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseSemanticCaptions(!!checked);
    };

    const onShouldStreamChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setShouldStream(!!checked);
    };

    const onExcludeCategoryChanged = (_ev?: React.FormEvent, newValue?: string) => {
        setExcludeCategory(newValue || "");
    };

    const onUseSuggestFollowupQuestionsChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseSuggestFollowupQuestions(!!checked);
    };

    const onUseOidSecurityFilterChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseOidSecurityFilter(!!checked);
    };

    const onUseGroupsSecurityFilterChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseGroupsSecurityFilter(!!checked);
    };

    const onExampleClicked = (example: string) => {
        makeApiRequest(example);
    };

    const onShowCitation = (citation: string, index: number) => {
        if (activeCitation === citation && activeAnalysisPanelTab === AnalysisPanelTabs.CitationTab && selectedAnswer === index) {
            setActiveAnalysisPanelTab(undefined);
        } else {
            setActiveCitation(citation);
            setActiveAnalysisPanelTab(AnalysisPanelTabs.CitationTab);
        }

        setSelectedAnswer(index);
    };

    const onToggleTab = (tab: AnalysisPanelTabs, index: number) => {
        if (activeAnalysisPanelTab === tab && selectedAnswer === index) {
            setActiveAnalysisPanelTab(undefined);
        } else {
            setActiveAnalysisPanelTab(tab);
        }

        setSelectedAnswer(index);
    };

    return (
        <div className={styles.container}>

            <>
                {/* Backdrop when sidebar is open on mobile */}
                {isMobile && !collapsed && (
                    <div className={styles.backdrop} onClick={toggleCollapse}></div>
                )}
                {/* Sidebar: show if desktop or mobile and sidebar is open */}
                {(!isMobile || !collapsed) && (
                    <div className={`${styles.leftdiv} ${collapsed ? styles.collapsedLeftdiv : ""}`}>
                        <div className={`${styles.leftPanel} ${collapsed ? styles.collapsed : ""}`}>
                            <div className={collapsed ? styles.stack : styles.row}>
                                <div onClick={toggleCollapse} className={styles.iconWrapper}>
                                    <img src={CollapseIcon} alt="Toggle" style={{ width: '15px', height: '15px' }} />
                                </div>
                                <div className={collapsed ? styles.buttonWrapper : styles.buttonRight}>
                                    <ClearChatButton onClick={clearChat} disabled={isLoading} />
                                </div>
                            </div>

                            {!collapsed && (
                                <>
                                    <div className={styles.leftPanelTopContent}>
                                        <div className={styles.centeredContainer}></div>
                                        <div>
                                            <h3 className={styles.historyTitle}></h3>
                                            <div>
                                                {storedHistory.length === 0 ? (
                                                    <p>No chat history available.</p>
                                                ) : (
                                                    storedHistory.slice(-10).map((historyItem, index) => (
                                                        <div
                                                            key={index}
                                                            className={styles.historyItem}
                                                            onClick={() =>
                                                                handleHistoryClick(historyItem.question, historyItem.response)
                                                            }
                                                        >
                                                            <span className={styles.historyItemText}>
                                                                {historyItem.question.length > 25
                                                                    ? `${historyItem.question.slice(0, 25)}...`
                                                                    : historyItem.question}
                                                            </span>
                                                        </div>
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className={styles.leftPanelBottomButtonsContainer}>
                                            <a
                                                href="https://app-backend-yfktjr4j7kgpw.azurewebsites.net/"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className={styles.leftPanelButtonWrapper}
                                                style={{
                                                    textDecoration: "none",      
                                                    display: "inline-block"       
                                                }}
                                            >
                                                <button 
                                                    className={`${styles.leftPanelButton} ${styles.camStar}`}
                                                    style={{
                                                    width : "100%",
                                                    backgroundColor: "#ffffff",             
                                                    border: "1px solid #ccc",              // Light gray border
                                                    borderRadius: "4px",                   // Rounded corners
                                                    padding: "8px 20px",                   // Increase the “white box” around the text
                                                    fontSize: "12px",                      // Slightly larger font
                                                    display: "inline-flex",                // So icon + text stay centered
                                                    alignItems: "center",
                                                    textDecoration: "none",                // Ensure button text has no underline
                                                    cursor: "pointer",                      // Show pointer on hover                   
                                                    }}
                                                >
                                                <span className={styles.buttonIcon}>
                                                    <img src={MonitorIcon} style={{ width: '10px', height: '10px' }} />
                                                </span >
                                                UTAC ChatBot
                                                </button>
                                            </a>
                                            

                                        {/* <button className={`${styles.leftPanelButton} ${styles.operation}`}>
                                            <span className={styles.buttonIcon}>
                                                <img src={MonitorIcon} style={{ width: '10px', height: '10px' }} />
                                            </span>
                                            Operation AI Assistant
                                        </button> */}

                                        <div className={styles.userGuideContainer}>
                                            <InfoRegular className={styles.infoIcon} />
                                            <span>User Guide</span>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                )}
            </>



            <div className={`${styles.chatRoot} ${collapsed ? styles.expandedChatRoot : ""}`}>
                <div className={styles.chatContainer}>
                    <div className={styles.headerFlex}>
                        {/* Floating toggle button on mobile when sidebar is collapsed */}
                        {collapsed && (
                            <div className={styles.floatingControls}>
                                <div onClick={toggleCollapse} className={styles.iconWrapperOut}>
                                    <img src={CollapseIcon} alt="Open Sidebar" style={{ width: '15px', height: '15px' }} />
                                </div>
                                <div className={styles.buttonWrapperOut}>
                                    <ClearChatButton onClick={clearChat} disabled={isLoading} />
                                </div>
                            </div>
                        )}
                        <div className={styles.headerContainer}>
                            <div className={styles.chatbotTitle}>
                                <img src={UTACLogo} className={styles.logoHeader} /><span><h2>IT Helpdesk AI ChatBOT</h2></span>
                            </div>

                            <div className={`${styles.rightContainer} `}>

                                <select
                                    value={selectedCountry}
                                    onChange={(e) => {
                                        const newCountry = e.target.value as Country;
                                        setSelectedCountry(newCountry);
                                    }}
                                    className={styles.roundedDropdown}
                                >
                                    {countryOptions.map((option) => (
                                        <option key={option.key} value={option.key}>
                                            {option.text}
                                        </option>
                                    ))}
                                </select>
                                <select
                                    value={selectedSite}
                                    onChange={(e) => {
                                        const newSite = e.target.value;
                                        setSelectedSite(newSite);
                                    }}

                                    className={styles.roundedDropdown}
                                >
                                    {filteredSiteOptions.map((option) => (
                                        <option key={option.key} value={option.key}>
                                            {option.text}
                                        </option>
                                    ))}
                                </select>

                            </div>
                        </div>
                    </div>
                    {isFirstChat && (
                        <div className={styles.greetingMessage}>
                            <h2>Hello, {instance.getActiveAccount()?.username || appServicesToken?.user_claims?.name}</h2>
                            <h5>How can I help you today?</h5>
                        </div>
                    )}

                    <div className={styles.chatMessageStream}>
                        {isStreaming &&
                            streamedAnswers.map((streamedAnswer, index) => (
                                <div key={index}>
                                    <UserChatMessage message={streamedAnswer[0]} />
                                    <div className={styles.chatMessageGpt}>
                                        <Answer
                                            isStreaming={true}
                                            key={index}
                                            answer={streamedAnswer[1]}
                                            isSelected={false}
                                            onCitationClicked={c => onShowCitation(c, index)}
                                            onThoughtProcessClicked={() => onToggleTab(AnalysisPanelTabs.ThoughtProcessTab, index)}
                                            onSupportingContentClicked={() => onToggleTab(AnalysisPanelTabs.SupportingContentTab, index)}
                                            onLikeClicked={() => handleLike(index)}
                                            onDislikeClicked={() => {
                                                setLikeStatus("dislike");
                                                setShowDislikePopup(true);
                                            }}
                                            onFollowupQuestionClicked={q => makeApiRequest(q)}
                                            showFollowupQuestions={useSuggestFollowupQuestions && answers.length - 1 === index}
                                            lastAnswer={answers.length - 1}
                                        />
                                        {showDislikePopup && (
                                            <DislikePopup
                                                isOpen={showDislikePopup}
                                                onClose={() => setShowDislikePopup(false)}
                                                onSubmit={async (reason: string) => {
                                                    setDislikeReason(reason);
                                                    setShowDislikePopup(false);
                                                    const userId = appServicesToken != null ? appServicesToken.user_id : undefined;
                                                    const token = client ? await getToken(client) : undefined;
                                                    const request: SaveRequest = {
                                                        question: answers[selectedAnswer][0],
                                                        answer: answers[selectedAnswer][1].choices[0].message.content,
                                                        user_id: userId || undefined,
                                                        country: selectedCountry,
                                                        site: selectedSite,
                                                        metadata: { likeStatus: "dislike", dislikeReason: reason }
                                                    };
                                                    saveApi(request, token);
                                                }}
                                            />
                                        )}
                                    </div>
                                </div>
                            ))}
                        {!isStreaming &&
                            (selectedHistory && hasLoaded ? (
                                // If there's a selected history, show only the selected history
                                <div key={selectedHistory.question}>
                                    <UserChatMessage message={selectedHistory.question} />
                                    <div className={styles.chatMessageGpt}>
                                        <Answer
                                            isStreaming={false}
                                            key={selectedHistory.question}
                                            answer={selectedHistory.response}
                                            isSelected={true}
                                            onCitationClicked={(c) => onShowCitation(c, 0)}
                                            onThoughtProcessClicked={() => onToggleTab(AnalysisPanelTabs.ThoughtProcessTab, 0)}
                                            onSupportingContentClicked={() => onToggleTab(AnalysisPanelTabs.SupportingContentTab, 0)}
                                            onLikeClicked={() => handleLike(0)}
                                            onDislikeClicked={() => {
                                                setLikeStatus("dislike");
                                                setShowDislikePopup(true);
                                            }}
                                            onFollowupQuestionClicked={(q) => makeApiRequest(q)}
                                            showFollowupQuestions={false}
                                            lastAnswer={-1} // Update this as per your logic
                                        />
                                        {showDislikePopup && (
                                            <DislikePopup
                                                isOpen={showDislikePopup}
                                                onClose={() => setShowDislikePopup(false)}
                                                onSubmit={async (reason) => {
                                                    setDislikeReason(reason);
                                                    setShowDislikePopup(false);
                                                    const userId = appServicesToken != null ? appServicesToken.user_id : undefined;
                                                    const token = client ? await getToken(client) : undefined;
                                                    const request: SaveRequest = {
                                                        question: selectedHistory.question,
                                                        answer: selectedHistory.response.choices[0].message.content,
                                                        user_id: userId || undefined,
                                                        country: selectedCountry,
                                                        site: selectedSite,
                                                        metadata: { likeStatus: "dislike", dislikeReason: reason }
                                                    };
                                                    saveApi(request, token);
                                                }}
                                            />
                                        )}
                                    </div>
                                </div>
                            ) : (
                                answers.map((answer, index) => (
                                    <div key={index}>
                                        <UserChatMessage message={answer[0]} />
                                        <div className={styles.chatMessageGpt}>
                                            <Answer
                                                isStreaming={false}
                                                key={index}
                                                answer={answer[1]}
                                                isSelected={selectedAnswer === index && activeAnalysisPanelTab !== undefined}
                                                onCitationClicked={c => onShowCitation(c, index)}
                                                onThoughtProcessClicked={() => onToggleTab(AnalysisPanelTabs.ThoughtProcessTab, index)}
                                                onSupportingContentClicked={() => onToggleTab(AnalysisPanelTabs.SupportingContentTab, index)}
                                                onLikeClicked={() => handleLike(index)}
                                                onDislikeClicked={() => {
                                                    setLikeStatus("dislike");
                                                    setShowDislikePopup(true);
                                                }}
                                                onFollowupQuestionClicked={q => makeApiRequest(q)}
                                                showFollowupQuestions={useSuggestFollowupQuestions && answers.length - 1 === index}
                                                lastAnswer={answers.length - 1}
                                            />
                                            {showDislikePopup && (
                                                <DislikePopup
                                                    isOpen={showDislikePopup}
                                                    onClose={() => setShowDislikePopup(false)}
                                                    onSubmit={async reason => {
                                                        setDislikeReason(reason);
                                                        setShowDislikePopup(false);
                                                        const userId = appServicesToken != null ? appServicesToken.user_id : undefined;
                                                        const token = client ? await getToken(client) : undefined;
                                                        const request: SaveRequest = {
                                                            question: answers[selectedAnswer][0],
                                                            answer: answers[selectedAnswer][1].choices[0].message.content,
                                                            user_id: userId || undefined,
                                                            country: selectedCountry,
                                                            site: selectedSite,
                                                            metadata: { likeStatus: "dislike", dislikeReason: reason }
                                                        };
                                                        saveApi(request, token);
                                                    }}
                                                />
                                            )}
                                        </div>
                                    </div>
                                ))
                            ))
                        }

                        {isLoading && (
                            <>
                                <UserChatMessage message={lastQuestionRef.current} />
                                <div className={styles.chatMessageGptMinWidth}>
                                    <AnswerLoading />
                                </div>
                            </>
                        )}
                        {error ? (
                            <>
                                <UserChatMessage message={lastQuestionRef.current} />
                                <div className={styles.chatMessageGptMinWidth}>
                                    <AnswerError error={error.toString()} onRetry={() => makeApiRequest(lastQuestionRef.current)} />
                                </div>
                            </>
                        ) : null}
                        <div ref={chatMessageStreamEnd} />
                    </div>


                    <div className={styles.chatInput}>
                        {/* <div className={styles.preTextInput}>
                            By using the UTAC Chatbot, you acknowledge that you have read, understood, and agree to abide by these{" "}
                            <a href="https://utacgroupsg.sharepoint.com/sites/UTACPortal/SitePages/CustomPages/Our-IT-Policy-TermsOfUse.aspx" target="_blank">
                                Terms of Use.
                            </a>
                        </div> */}
                        {/* <div className={styles.centeredContainer}>
                            <ClearChatButton
                                onClick={clearChat}
                                disabled={!lastQuestionRef.current || isLoading}
                            />
                        </div> */}

                        <QuestionInput
                            clearOnSend
                            placeholder="Ask anything"
                            disabled={isLoading || isStreaming}
                            onSend={question => makeApiRequest(question)}
                        //onCountryChange={(country) => console.log("Country changed:", country)}
                        //onSiteChange={(site) => console.log("Site changed:", site)}
                        />

                        <div className={styles.postTextInput}>
                            <a target="_blank" href="">
                                Term of Use
                            </a>
                        </div>
                    </div>
                </div>

                {answers.length > 0 && activeAnalysisPanelTab && (
                    <AnalysisPanel
                        className={styles.chatAnalysisPanel}
                        activeCitation={activeCitation}
                        onActiveTabChanged={x => onToggleTab(x, selectedAnswer)}
                        citationHeight="810px"
                        answer={answers[selectedAnswer][1]}
                        activeTab={activeAnalysisPanelTab}
                    />
                )}

                <Panel
                    headerText="Configure answer generation"
                    isOpen={isConfigPanelOpen}
                    isBlocking={false}
                    onDismiss={() => setIsConfigPanelOpen(false)}
                    closeButtonAriaLabel="Close"
                    onRenderFooterContent={() => <DefaultButton onClick={() => setIsConfigPanelOpen(false)}>Close</DefaultButton>}
                    isFooterAtBottom={true}
                >
                    <TextField
                        className={styles.chatSettingsSeparator}
                        defaultValue={promptTemplate}
                        label="Override prompt template"
                        multiline
                        autoAdjustHeight
                        onChange={onPromptTemplateChange}
                    />

                    <SpinButton
                        className={styles.chatSettingsSeparator}
                        label="Retrieve this many search results:"
                        min={1}
                        max={50}
                        defaultValue={retrieveCount.toString()}
                        onChange={onRetrieveCountChange}
                    />
                    <TextField className={styles.chatSettingsSeparator} label="Exclude category" onChange={onExcludeCategoryChanged} />
                    <Checkbox
                        className={styles.chatSettingsSeparator}
                        checked={useSemanticRanker}
                        label="Use semantic ranker for retrieval"
                        onChange={onUseSemanticRankerChange}
                    />
                    <Checkbox
                        className={styles.chatSettingsSeparator}
                        checked={useSemanticCaptions}
                        label="Use query-contextual summaries instead of whole documents"
                        onChange={onUseSemanticCaptionsChange}
                        disabled={!useSemanticRanker}
                    />
                    <Checkbox
                        className={styles.chatSettingsSeparator}
                        checked={useSuggestFollowupQuestions}
                        label="Suggest follow-up questions"
                        onChange={onUseSuggestFollowupQuestionsChange}
                    />

                    {showGPT4VOptions && (
                        <GPT4VSettings
                            gpt4vInputs={gpt4vInput}
                            isUseGPT4V={useGPT4V}
                            updateUseGPT4V={useGPT4V => {
                                setUseGPT4V(useGPT4V);
                            }}
                            updateGPT4VInputs={inputs => setGPT4VInput(inputs)}
                        />
                    )}

                    <VectorSettings
                        showImageOptions={useGPT4V && showGPT4VOptions}
                        updateVectorFields={(options: VectorFieldOptions[]) => setVectorFieldList(options)}
                        updateRetrievalMode={(retrievalMode: RetrievalMode) => setRetrievalMode(retrievalMode)}
                    />

                    {useLogin && (
                        <Checkbox
                            className={styles.chatSettingsSeparator}
                            checked={useOidSecurityFilter || requireAccessControl}
                            label="Use oid security filter"
                            disabled={!isLoggedIn(client) || requireAccessControl}
                            onChange={onUseOidSecurityFilterChange}
                        />
                    )}
                    {useLogin && (
                        <Checkbox
                            className={styles.chatSettingsSeparator}
                            checked={useGroupsSecurityFilter || requireAccessControl}
                            label="Use groups security filter"
                            disabled={!isLoggedIn(client) || requireAccessControl}
                            onChange={onUseGroupsSecurityFilterChange}
                        />
                    )}

                    <Checkbox
                        className={styles.chatSettingsSeparator}
                        checked={shouldStream}
                        label="Stream chat completion responses"
                        onChange={onShouldStreamChange}
                    />
                    {useLogin && <TokenClaimsDisplay />}
                </Panel>
            </div>
        </div>
    );
};

export default Chat;
