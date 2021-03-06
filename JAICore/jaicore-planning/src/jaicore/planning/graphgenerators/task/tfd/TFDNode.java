package jaicore.planning.graphgenerators.task.tfd;

import java.io.Serializable;
import java.util.List;

import jaicore.logic.fol.structure.Literal;
import jaicore.logic.fol.structure.Monom;
import jaicore.planning.model.core.Action;
import jaicore.planning.model.task.stn.MethodInstance;

public class TFDNode implements Serializable {

	private static final long serialVersionUID = 7710905829501897491L;

	private static long maxID = 0L;
	private long ID = maxID++;
	public long getID() {
		return ID;
	}

	private TFDRestProblem problem;
	private final MethodInstance appliedMethodInstance;
	private final Action appliedAction;
	private final boolean isGoal;
	
	public TFDNode(Monom initialState, List<Literal> remainingTasks) {
		this (initialState, remainingTasks, null, null);
	}
	
	public TFDNode(MethodInstance appliedMethodInstance, boolean isGoal) {
		super();
		this.problem = null;
		this.appliedMethodInstance = appliedMethodInstance;
		this.appliedAction = null;
		this.isGoal = isGoal;
	}

	public TFDNode(Action appliedAction, boolean isGoal) {
		super();
		this.problem = null;
		this.appliedAction = appliedAction;
		this.appliedMethodInstance = null;
		this.isGoal = isGoal;
	}
	
	public TFDNode(Monom state, List<Literal> remainingTasks, MethodInstance appliedMethodInstance, Action appliedAction) {
		super();
		this.problem = new TFDRestProblem(state, remainingTasks);
		this.appliedMethodInstance = appliedMethodInstance;
		this.appliedAction = appliedAction;
		this.isGoal = remainingTasks.isEmpty();
	}
	
	public TFDRestProblem getProblem() {
		return problem;
	}

	public Monom getState() {
		return problem.getState();
	}

	public List<Literal> getRemainingTasks() {
		return problem.getRemainingTasks();
	}

	public Action getAppliedAction() {
		return appliedAction;
	}
	

	public MethodInstance getAppliedMethodInstance() {
		return appliedMethodInstance;
	}

	public boolean isGoal() {
		return isGoal;
	}
	
	public void clear() {
		this.problem = null;
	}

	@Override
	public String toString() {
		return "TFDNode:"  + ID;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + (int) (ID ^ (ID >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		TFDNode other = (TFDNode) obj;
		if (ID != other.ID)
			return false;
		return true;
	}
}
