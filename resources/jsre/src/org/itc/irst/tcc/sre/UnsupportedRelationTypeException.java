/*
 * Copyright 2005 FBK-irst (http://www.fbk.eu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 package org.itc.irst.tcc.sre;

/**
 * Thrown to indicate that the instance kernel
 * cannot be loaded and instantiated.
 *
 * @author 	Claudio Giuliano
 * @version 1.0
 * @since		1.0
 */
class UnsupportedRelationTypeException extends ClassNotFoundException
{
	/**
	 * Constructs a <code>UnsupportedRelationTypeException</code>
	 * with no detail message.
	 */
	public UnsupportedRelationTypeException()
	{
		super();
	} // end constructor

	/**
	 * Constructs a <code>UnsupportedRelationTypeException</code>
	 * with the specified detail message. 
	 *
	 * @param s	the detail message.
	 */
	public UnsupportedRelationTypeException(String s)
	{
		super(s);
	} // end constructor
	
} // end class UnsupportedRelationTypeException